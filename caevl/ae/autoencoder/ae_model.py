import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timm.layers import trunc_normal_

from caevl.ae.autoencoder.ae_blocks import Encoder, Decoder
from caevl.ae.autoencoder.perceptual_loss import Perceptual_DinoV2


class AutoEncoder(torch.nn.Module):
    '''
    Auto Encoder.
    '''

    def __init__(self,
                 input_dimensions: tuple,
                 nb_input_channels: int,
                 nb_channels_fst_feature_map: int,
                 latent_dim: int,
                 learning_rate: float,
                 scheduler: bool,
                 loss_name: str,
                 canny_edges: bool = False,
                 device: str = 'auto',
                 alpha: float = 1,
                 flag_perceptual_loss: bool = False,
                 scalar_divide_channels_fst_feature_map_for_decoder: int = 1,
                 encoder_only: bool = False,
                 softmax: bool = False,
                 sigmoid: bool = False,
                 use_cbam: bool = False,
                 return_decoded_in_forward=True) -> None:
        """
        Initialize Encoder and Decoder of the AutoEncoder.
        Initialize optimizer and scheduler.

        Parameters
        ----------
        input_dimensions : tuple (int, int)
            (height, width) of the input data.
        nb_input_channels : int
            Number of channels of the input data.
        nb_channels_fst_feature_map : int
            Number of channels of the first layer. of the encoder.
        latent_dim : int
            Dimension of the latent space.
        learning_rate : float
            Initial value of learning rate.
        scheduler : bool
            If True, optimizer is coupled with a scheduler.
        loss_name : str
            Name of the loss function to use, either 'L1' or 'L2'
        canny_edges : bool, optional
            If True, means the autoencoder is fed with binary Canny images.
            In that case, the last layer of the decoder is activated with a
            sigmoid, by default False.
        device : str, optional
            Device on which to train the model, by default 'auto'.
            If 'auto', checks whether cuda is available.
        alpha : float, optional
            Scalar for the perceptual loss, by default 1.
        flag_perceptual_loss : bool, optional
            Whether to add perceptual loss to training loss, by default False.
        scalar_divide_channels_fst_feature_map_for_decoder : int, optional
            The first feature map of the decoder
            has nb_channels_fst_feature_map // scalar_divide_channels_fst_feature_map_for_decoder
            channels, by default 1.
        encoder_only : bool, optional
            If True, only the encoder is used, by default False.
        softmax : bool, optional
            If True, use softmax activation in the last layer of the decoder, by default False.
        sigmoid : bool, optional
            If True, use sigmoid activation in the last layer of the decoder, by default False.
        use_cbam : bool, optional
            If True, use CBAM in the perceptual loss to pass image from c channels to 3 before DinoV2, by default False.
        """

        super(AutoEncoder, self).__init__()

        # self variables
        self.input_dimensions = input_dimensions
        self.nb_input_channels = nb_input_channels
        self.nb_channels_fst_feature_map = nb_channels_fst_feature_map
        self.latent_dim = latent_dim

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.flag_perceptual_loss = flag_perceptual_loss
        self.use_cbam = use_cbam

        self.loss_name = loss_name
        grayscale_shift = self.nb_input_channels == 1
        self.perceptual_loss = Perceptual_DinoV2(device=self.device,
                                                 grayscale_shift=grayscale_shift,
                                                 use_cbam=use_cbam,
                                                 cbam_channels=[self.nb_input_channels, 6, 3]) if flag_perceptual_loss else None
        self.alpha = alpha

        # loss function
        if self.loss_name == 'L2':
            self.loss_fun = nn.MSELoss(reduction='mean')
        else:
            self.loss_fun = nn.L1Loss(reduction='mean')

        # initialize encoder
        self.encoder = Encoder(self.nb_input_channels,
                               self.nb_channels_fst_feature_map,
                               self.latent_dim)

        # initialize decoder
        self.nb_blocks = 5
        if isinstance(self.input_dimensions, int):
            self.input_dimensions = (self.input_dimensions, self.input_dimensions)
        self.last_feature_size = (self.input_dimensions[0]//2**(self.nb_blocks-1), self.input_dimensions[1]//2**(self.nb_blocks-1))

        self.scalar_divide_channels_fst_feature_map_for_decoder = scalar_divide_channels_fst_feature_map_for_decoder
        nb_channels_fst_feature_map_decoder = self.nb_channels_fst_feature_map // scalar_divide_channels_fst_feature_map_for_decoder

        self.encoder_only = encoder_only
        if self.encoder_only:
            self.decoder = nn.Identity()
        else:
            self.decoder = Decoder(self.nb_input_channels,
                                   nb_channels_fst_feature_map_decoder,
                                   self.latent_dim,
                                   self.last_feature_size,
                                   canny_edges,
                                   softmax,
                                   sigmoid)

        self.initialize_weights()

        # initialize optimizer
        self.learning_rate = learning_rate
        self.optimizer = AdamW(self.parameters(), lr=learning_rate)

        # initialize scheduler
        self.flag_scheduler = scheduler
        if scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                               factor=0.5, patience=3)
        else:
            self.scheduler = None

        self.return_decoded_in_forward = return_decoded_in_forward

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and not isinstance(m, nn.LazyLinear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):        
        self.apply(self._init_weights)

    def loss_function(self, inputs, outputs, embeddings=None, coordinates=None):
        loss = 0

        ### PIXEL-WISE LOSS ###
        if not self.encoder_only:
            loss += self.loss_fun(inputs, outputs)

        ### PERCEPTUAL LOSS ###
        if self.flag_perceptual_loss:
            b, c, h, w = inputs.shape
            if c > 3 and not self.use_cbam:
                inputs = torch.argmax(inputs, dim=1).to(inputs.dtype) / self.nb_input_channels
                outputs = torch.argmax(outputs, dim=1).to(outputs.dtype) / self.nb_input_channels
            perceptual_loss = self.perceptual_loss(inputs, outputs)
            loss += self.alpha * perceptual_loss

        return loss

    def forward(self, x):
        # encode
        embedding = self.encoder(x)
        # decode
        decoded = self.decoder(embedding)

        if not self.return_decoded_in_forward:
            return embedding
        return decoded, embedding

    def forward_encoder(self, x):
        embedding = self.encoder(x)
        return embedding

    def reset_optimizer(self, learning_rate=None):
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate
        self.optimizer = AdamW(self.parameters(), lr=learning_rate)

        if self.flag_scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                               factor=0.5, patience=3)
        else:
            self.scheduler = None
