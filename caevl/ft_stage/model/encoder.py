from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn


class CaevlFT(nn.Module):
    def __init__(self, backbone, projector, local_projector, device,
                 invariance_coeff, std_coeff, cov_coeff, alpha,
                 l2_all_matches=True, input_dimensions=None):
        super(CaevlFT, self).__init__()

        self.criterion = nn.MSELoss(reduction='none')
        self.epsilon = 1e-5
        self.gamma = 1
        self.alpha = alpha
        self.l2_all_matches = l2_all_matches
        self.num_matches = (20, 4)

        self.backbone = backbone
        if self.alpha > 0:
            self.projector = projector
        if self.alpha < 1:
            self.local_projector = local_projector
        self.device = device

        self.invariance_coeff = invariance_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

        if input_dimensions is not None:
            h, w = input_dimensions
            self.ign_location_map = (0, 0, h, w, h, w)
            self.ign_location_map = _location_to_NxN_grid(self.ign_location_map, N=7).reshape(49, 2)

    def train_mode(self, mode=True):
        r"""Sets the module in training mode."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval_mode(self):
        return self.train_mode(mode=False)

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _loss_embeddings(self, x, y):
        invariance_loss = self.criterion(x, y).mean(dim=-1)  # shape (batch_size,)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + self.epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(self.gamma - std_x)) / 2 + torch.mean(F.relu(self.gamma - std_y)) / 2
        std_loss = std_loss.repeat(x.shape[0])

        batch_size = x.shape[0]
        num_features = x.shape[-1]

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = self._off_diagonal(cov_x).pow_(2).sum().div(num_features)
        cov_loss += self._off_diagonal(cov_y).pow_(2).sum().div(num_features)
        cov_loss = cov_loss.repeat(x.shape[0])

        loss = self.invariance_coeff * invariance_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

        return loss

    def _loss_maps(self, x, y):

        invariance_loss = self.invariance_coeff * self.criterion(x, y).mean(dim=(-2, -1))

        std_x = torch.sqrt(x.var(dim=0) + self.epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.epsilon)
        std_loss = self.std_coeff * (
            torch.mean(F.relu(self.gamma - std_x)) / 2 + torch.mean(F.relu(self.gamma - std_y)) / 2
        )
        std_loss = std_loss.repeat(x.shape[0])

        x = x.permute((1, 0, 2))  # shape b, (h*w), c -> (h*w), b, c
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = BC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...bc,...bd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...bc,...bd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss.repeat(x.shape[0])
        cov_loss = self.cov_coeff * cov_loss

        return invariance_loss, std_loss, cov_loss

    def _local_loss(
        self, i, j, maps_1, maps_2, locations
    ):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        maps_1 = rearrange(maps_1, 'b c h w -> b (h w) c')
        maps_2 = rearrange(maps_2, 'b c h w -> b (h w) c')

        # Feature-based matching
        if self.l2_all_matches:
            num_matches_on_l2 = [None, None]
        else:
            num_matches_on_l2 = self.num_matches

        maps_1_filtered, maps_1_nn = nn_on_l2(
            maps_1, maps_2, num_matches=num_matches_on_l2[0]
        )
        maps_2_filtered, maps_2_nn = nn_on_l2(
            maps_2, maps_1, num_matches=num_matches_on_l2[1]
        )

        inv_loss_1, var_loss_1, cov_loss_1 = self._loss_maps(maps_1_filtered, maps_1_nn)
        inv_loss_2, var_loss_2, cov_loss_2 = self._loss_maps(maps_2_filtered, maps_2_nn)
        var_loss += (var_loss_1 / 2 + var_loss_2 / 2)
        cov_loss += (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss += (inv_loss_1 / 2 + inv_loss_2 / 2)

        # Location based matching
        if locations is not None:
            h, w = locations[0, -2:]
            location_1 = self.ign_location_map.unsqueeze(0).repeat(locations.shape[0], 1, 1).to(torch.float).to(locations.device)
            location_2 = locations.to(torch.float)

            maps_1_filtered, maps_1_nn = nn_on_location(
                location_1,
                location_2,
                maps_1,
                maps_2,
                num_matches=self.num_matches[0],
            )
            maps_2_filtered, maps_2_nn = nn_on_location(
                location_2,
                location_1,
                maps_2,
                maps_1,
                num_matches=self.num_matches[1],
            )

            inv_loss_1, var_loss_1, cov_loss_1 = self._loss_maps(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._loss_maps(maps_2_filtered, maps_2_nn)
            var_loss += (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss += (cov_loss_1 / 2 + cov_loss_2 / 2)

            inv_loss += (inv_loss_1 / 2 + inv_loss_2 / 2)

        return inv_loss, var_loss, cov_loss

    def local_loss(self, maps_1, maps_2, locations=None):

        inv_loss, var_loss, cov_loss = self._local_loss(
            0, 1, maps_1, maps_2, locations,
        )

        return inv_loss + var_loss + cov_loss

    def forward(self, x, y, locations=None):
        with torch.enable_grad():
            unpooled_x, pooled_x = self.backbone.forward_features_unpooled(x)
            unpooled_y, pooled_y = self.backbone.forward_features_unpooled(y)

            device = pooled_x.device

            ### GLOBAL LOSS ###
            if self.alpha == 0:
                global_loss = torch.tensor(0.0, device=device)
            else:
                projected_x = self.projector(pooled_x)
                projected_y = self.projector(pooled_y)

                global_loss = self._loss_embeddings(projected_x, projected_y)

            ### LOCAL LOSS ###
            if self.alpha == 1:
                local_loss = torch.tensor(0.0, device=device)
            else:
                b, c, h, w = unpooled_x.shape

                local_projected_x = self.local_projector(unpooled_x.flatten(start_dim=1))
                local_projected_y = self.local_projector(unpooled_y.flatten(start_dim=1))

                local_projected_x = local_projected_x.view(b, -1, h, w)
                local_projected_y = local_projected_y.view(b, -1, h, w)

                local_loss = self.local_loss(local_projected_x, local_projected_y, locations=locations)

            return self.alpha * global_loss + (1 - self.alpha) * local_loss


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def nearest_neighbors(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0)

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )

    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )

    indices_selected = indices.masked_select(mask)
    indices_selected = indices_selected.reshape(batch_size, num_matches)

    filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    return filtered_input_maps, filtered_candidate_maps


def nn_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_maps, candidate_maps)
    return nearest_neighbors(input_maps, candidate_maps, distances, num_matches)


def nn_on_location(
    input_location, candidate_location, input_maps, candidate_maps, num_matches
):
    """
    input_location: (B, H * W, 2)
    candidate_location: (B, H * W, 2)
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """

    distances = torch.cdist(input_location, candidate_location)
    return nearest_neighbors(input_maps, candidate_maps, distances, num_matches)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def _location_to_NxN_grid(location, N=8):
    i, j, h, w, H, W = location
    size_h_case = h / N
    size_w_case = w / N
    half_size_h_case = size_h_case / 2
    half_size_w_case = size_w_case / 2
    final_grid_x = torch.zeros(N, N)
    final_grid_y = torch.zeros(N, N)

    final_grid_x[0][0] = i + half_size_h_case
    final_grid_y[0][0] = j + half_size_w_case
    for k in range(1, N):
        final_grid_x[k][0] = final_grid_x[k - 1][0] + size_h_case
        final_grid_y[k][0] = final_grid_y[k - 1][0]
    for l in range(1, N):
        final_grid_x[0][l] = final_grid_x[0][l - 1]
        final_grid_y[0][l] = final_grid_y[0][l - 1] + size_w_case
    for k in range(1, N):
        for l in range(1, N):
            final_grid_x[k][l] = final_grid_x[k - 1][l] + size_h_case
            final_grid_y[k][l] = final_grid_y[k][l - 1] + size_w_case

    final_grid = torch.stack([final_grid_x, final_grid_y], dim=-1)

    return final_grid
