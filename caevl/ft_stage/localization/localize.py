import numpy as np
from tqdm import tqdm

import torch


def get_embeddings_and_coords(dataloader, model, record_gradients=False):
    embeddings = torch.zeros((len(dataloader.dataset), model.backbone.latent_dim), dtype=torch.float32)
    coordinates = torch.zeros((len(dataloader.dataset), 2))
    indices = torch.zeros((len(dataloader.dataset),), dtype=torch.int)
    current_index = 0

    gradients = torch.enable_grad() if record_gradients else torch.inference_mode()

    with gradients:
        with tqdm(total=len(dataloader)) as pbar:
            for elements in iter(dataloader):
                data = elements[0]
                coords = elements[2]
                idx = elements[3]
                batch_size = data.shape[0]
                data = data.to(model.device)

                _, embeddings_batch = model.backbone.forward_features_unpooled(data, training=False)
                embeddings_batch = embeddings_batch.to(torch.float32)

                embeddings[current_index: current_index + batch_size] = embeddings_batch
                coordinates[current_index: current_index + batch_size] = coords
                indices[current_index: current_index + batch_size] = idx
                current_index += batch_size
                pbar.update(1)
    return embeddings, coordinates, indices


# def localize_batch(data, coords_to_loc, idx_to_localize, vicreg, images_to_localize_loader,
#                    reference_images_loader, reference_embeddings, coords_ref,
#                    nb_scores_to_keep, current_index, all_scores, all_kept_indices, dict_all_preds,
#                    distance_criteria, top_k_accuracies, accuracies):
#     data = data.to(vicreg.device)
#     _, embeddings_batch = vicreg.backbone.forward_features_unpooled(data, training=False)
#     embeddings_batch = embeddings_batch.to(torch.float32)
#     embeddings_batch /= torch.norm(embeddings_batch, dim=-1)[:, None]

#     # similarity scores
#     scores_batch = torch.matmul(embeddings_batch, reference_embeddings.T)

#     batch_size = data.shape[0]

#     # Compute top-k once (the largest k you need)
#     max_k = len(reference_images_loader.dataset)
#     top_k_scores = torch.topk(scores_batch, k=max_k, dim=1)

#     topk_values = top_k_scores.values
#     top_k_indices = top_k_scores.indices.cpu()

#     keep_values = topk_values[:, :nb_scores_to_keep].cpu().numpy().astype(np.float16)
#     keep_indices = top_k_indices[:, :nb_scores_to_keep].cpu().numpy().astype(np.int32)

#     # Store into output arrays
#     sl = slice(current_index, current_index + batch_size)
#     all_scores[sl] = keep_values
#     all_kept_indices[sl] = keep_indices

#     # Extract the coordinates for both sets of images
#     x_to_loc, y_to_loc = coords_to_loc.cpu().numpy()[:, 0], coords_to_loc.cpu().numpy()[:, 1]

#     # coords_match = coords_ref[indices_ref[top_k_indices]]
#     coords_match = coords_ref[top_k_indices]
#     x_match, y_match = coords_match[:, :, 0], coords_match[:, :, 1]

#     # Compute the distances in a vectorized way:
#     distances = np.sqrt(np.array((x_match - x_to_loc[:, None])**2 + (y_match - y_to_loc[:, None])**2))

#     # smallest_distance
#     indices_smallest_distance = np.argmin(distances, axis=-1)
#     true_indices = [top_k_indices[i, indices_smallest_distance[i]] for i in range(len(indices_smallest_distance))]
#     scores_associated_to_smallest_distance = scores_batch[np.arange(batch_size), true_indices]
#     scores_associated_to_smallest_distance = scores_associated_to_smallest_distance.cpu().numpy()

#     # Create a dictionary mapping image names to their matches
#     dict_pred = dict()
#     image_names_to_loc = np.array(images_to_localize_loader.dataset.image_names)[np.array(idx_to_localize)]
#     for i, im in enumerate(image_names_to_loc):
#         ref_names = [
#             reference_images_loader.dataset.image_names[idx]
#             for idx in top_k_indices[i][:nb_scores_to_keep]
#         ]

#         ref_distances = distances[i][:nb_scores_to_keep].astype(np.float16)

#         best_match = (
#             indices_smallest_distance[i],
#             scores_associated_to_smallest_distance[i],
#         )

#         ref_scores = (
#             top_k_scores.values[i][:nb_scores_to_keep]
#             .cpu()
#             .numpy()
#             .astype(np.float16)
#         )

#         dict_pred[im] = [
#             ref_names,
#             ref_distances,
#             best_match,
#             ref_scores,
#         ]

#     if dict_all_preds is None:
#         dict_all_preds = dict_pred
#     else:
#         dict_all_preds.update(dict_pred)

#     for i, distance_criterion in enumerate(distance_criteria):
#         for j, top_k_accuracy in enumerate(top_k_accuracies):
#             accuracies[i][j] += np.any((distances[:, :top_k_accuracy] <= distance_criterion), axis=1).sum()

#     current_index += batch_size
#     return current_index, dict_all_preds


def localize_batch(data,
                   coords_to_loc,
                   idx_to_localize,
                   model,
                   images_to_localize_loader,
                   reference_images_loader,
                   reference_embeddings,
                   coords_ref,
                   nb_scores_to_keep,
                   current_index,
                   all_scores,
                   all_kept_indices,
                   dict_all_preds,
                   distance_criteria,
                   top_k_accuracies,
                   accuracies):

    data = data.to(model.device)
    _, embeddings_batch = model.backbone.forward_features_unpooled(data, training=False)

    embeddings_batch = embeddings_batch.to(torch.float32)
    embeddings_batch = torch.nn.functional.normalize(embeddings_batch, dim=-1)

    scores_batch = embeddings_batch @ reference_embeddings.T

    batch_size = data.shape[0]
    top_k_scores = torch.topk(scores_batch, k=nb_scores_to_keep, dim=1)
    topk_values = top_k_scores.values.cpu().numpy().astype(np.float16)
    topk_indices = top_k_scores.indices.cpu().numpy().astype(np.int32)

    # Store into output arrays
    sl = slice(current_index, current_index + batch_size)
    all_scores[sl] = topk_values
    all_kept_indices[sl] = topk_indices

    # ---------------------------
    # Distance computation
    # ---------------------------
    coords_to_loc_np = coords_to_loc.cpu().numpy()
    x_to_loc, y_to_loc = coords_to_loc_np[:, 0], coords_to_loc_np[:, 1]

    coords_match = coords_ref[topk_indices]  # shape (B, K, 2)
    x_match = coords_match[:, :, 0]
    y_match = coords_match[:, :, 1]

    distances = np.sqrt((x_match - x_to_loc[:, None])**2 + (y_match - y_to_loc[:, None])**2)

    # Smallest-distance evaluation
    indices_smallest_distance = np.argmin(distances, axis=1)
    true_indices = topk_indices[np.arange(batch_size), indices_smallest_distance]

    best_score = scores_batch[torch.arange(batch_size), true_indices]
    best_score = best_score.cpu().numpy()

    # ---------------------------------------------------------
    # Build per-image prediction dictionary
    # ---------------------------------------------------------
    if dict_all_preds is None:
        dict_all_preds = {}

    image_names_to_loc = np.array(images_to_localize_loader.dataset.image_names)[idx_to_localize]
    ref_names_list = reference_images_loader.dataset.image_names

    dict_pred = {}
    for i, im_name in enumerate(image_names_to_loc):

        ref_names = [ref_names_list[idx] for idx in topk_indices[i]]
        ref_distances = distances[i].astype(np.float16)
        best_match = (indices_smallest_distance[i], best_score[i])
        ref_scores = topk_values[i]  # already float16

        dict_pred[im_name] = [
            ref_names,
            ref_distances,
            best_match,
            ref_scores,
        ]

    dict_all_preds.update(dict_pred)

    # Update top-k accuracy metrics
    for i, dist_crit in enumerate(distance_criteria):
        for j, k in enumerate(top_k_accuracies):
            # is at least one of the top-k predictions within the threshold
            correct = np.any(distances[:, :k] <= dist_crit, axis=1)
            accuracies[i][j] += correct.sum()

    current_index += batch_size
    return current_index, dict_all_preds


def localize_per_batch(reference_images_loader, images_to_localize_loader, model, record_gradients=False):

    model.eval()
    reference_embeddings, coords_ref, indices_ref = get_embeddings_and_coords(reference_images_loader, model, record_gradients)
    reference_embeddings /= torch.norm(reference_embeddings, dim=-1)[:, None]
    reference_embeddings = reference_embeddings.to(model.device)

    nb_scores_to_keep = min(10, len(reference_embeddings))
    # only keep top scores to reduce memory consumption when saving
    all_scores = np.zeros((len(images_to_localize_loader.dataset), nb_scores_to_keep), dtype=np.float16)
    all_kept_indices = np.zeros((len(images_to_localize_loader.dataset), nb_scores_to_keep), dtype=np.int32)

    dict_all_preds = None
    distance_criteria = [15, 25, 50, 100, 150, 250, 500]
    top_k_accuracies = [1, 5, 10, 100]
    accuracies = np.zeros((len(distance_criteria), len(top_k_accuracies)))

    current_index = 0
    with torch.inference_mode():
        with tqdm(total=len(images_to_localize_loader)) as pbar:
            for data, _, coords_to_loc, idx_to_localize in images_to_localize_loader:

                current_index, dict_all_preds = localize_batch(data,
                                                               coords_to_loc,
                                                               idx_to_localize,
                                                               model,
                                                               images_to_localize_loader,
                                                               reference_images_loader,
                                                               reference_embeddings,
                                                               coords_ref,
                                                               nb_scores_to_keep,
                                                               current_index,
                                                               all_scores,
                                                               all_kept_indices,
                                                               dict_all_preds,
                                                               distance_criteria,
                                                               top_k_accuracies,
                                                               accuracies)

                pbar.set_postfix(acc5_at_100=accuracies[2][1] / current_index, acc5_at_500=accuracies[-1][1] / current_index)
                pbar.update(1)

    accuracies = np.array(accuracies) / len(images_to_localize_loader.dataset)

    dict_results = {
        'accuracies': accuracies,
        'dict_pred': dict_all_preds,
        'scores': all_scores.astype(np.float16),
    }

    return dict_results
