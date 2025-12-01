
import torch.nn as nn

from caevl.models.pretrained.dinov2.dinoV2 import DinoV2
from caevl.models.pretrained.CBAM.CBAM import CBAM_Bottleneck

class Perceptual_DinoV2(nn.Module):
    """
    Perceptual loss using DinoV2 model.

    Parameters
    ----------
    device : torch.device
        Device to run the model on (e.g., 'cuda' or 'cpu').
    grayscale_shift : bool
        Flag to indicate if grayscale shift should be applied.
        Grayscale shift takes a grayscale image and repeats it on 3 channels, but each channel is shifted by one pixel.
    use_cbam : bool, optional
        Flag to indicate if CBAM should be used. Default is False.
        CBAM is used to take an image from c channels to 3.
    cbam_channels : list, optional
        List of channels for CBAM bottleneck. Default is [12, 6, 3].
    """

    def __init__(self, device, grayscale_shift, 
                 use_cbam=False, cbam_channels=[12, 6, 3]):

        super(Perceptual_DinoV2, self).__init__()
        self.device = device

        self.dino = DinoV2(device, load_pretrained_weights=True, requires_grad=False, add_head=False,
                           grayscale_shift=grayscale_shift, no_grayscale=use_cbam)
        self.dino.eval()

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam_bottleneck = CBAM_Bottleneck(channels=cbam_channels, kernel_sizes=None, stride=1, bias=False, 
                                                   norm_layer='batch_norm', activation='relu')

    def forward(self, images, reconstructions):

        criterion = nn.MSELoss()
        loss = 0.
        images = self.dino.transform(images)
        reconstructions = self.dino.transform(reconstructions)

        if self.use_cbam:
            images = self.cbam_bottleneck.forward(images)
            reconstructions = self.cbam_bottleneck(reconstructions)

        images = self.dino.model.patch_embed(images)
        reconstructions = self.dino.model.patch_embed(reconstructions)

        nb_layers = 0
        for block in self.dino.model.blocks[0]:
            images = block(images)
            reconstructions = block(reconstructions)
            loss += criterion(images, reconstructions)
            nb_layers += 1

        loss /= nb_layers
        return loss
