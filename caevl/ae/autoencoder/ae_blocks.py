import torch
from torch import nn

class Encoder(torch.nn.Module):
    """Encoder part of the AutoEncoder."""
    
    def __init__(self,
                 nb_input_channels: int,
                 nb_channels_fst_feature_map: int,
                 latent_dim: int,
                 **kwargs) -> None:
        """Create an instance of an Encoder.

        Parameters
        ----------
        nb_input_channels : int
            Number of channels of the input data.
        nb_channels_fst_feature_map : int
            Number of channels of the first layer.
        latent_dim : int
            Dimension of the latent space.
        """
        super(Encoder, self).__init__()
        
        self.nb_input_channels = nb_input_channels
        self.nb_channels_fst_feature_map = nb_channels_fst_feature_map
        self.latent_dim = latent_dim
        
        modules = []
        #  Set the number of channels per layer
        hidden_dims = [self.nb_channels_fst_feature_map, 2*self.nb_channels_fst_feature_map, 
                       4*self.nb_channels_fst_feature_map, 8*self.nb_channels_fst_feature_map, 
                       8*self.nb_channels_fst_feature_map]

        ## BUILD ENCODER
        in_nb_features = nb_input_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_nb_features, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_nb_features = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.linear_bottleneck = torch.nn.LazyLinear(self.latent_dim)        
                
    def forward(self, x):

        y = self.encoder(x)

        y = torch.flatten(y, start_dim=1)
        
        # project to embedding space
        y = self.linear_bottleneck(y)
        return y
    
    def forward_features_unpooled(self, x, training=True):
        """Pass input data in the encoder, returning both unpooled and pooled representations.

        Parameters
        ----------
        x : torch.Tensor, shape (b, c, h, w)
            Input data.
        training : bool, optional
            Parameter to respect other model's 'forward_features_unpooled' function's parameters, not used here, by default True

        Returns
        -------
        tuple, (torch.Tensor, torch.Tensor), shapes (b, c_unpooled, h_unpooled, w_unpooled), (b, latent_dim)
            Data after the convolutional part of the encoder (unpooled) and after projection to latent space (pooled)
        """
        unpooled_x = self.encoder(x)
        pooled_x = torch.flatten(unpooled_x, start_dim=1)
        pooled_x = self.linear_bottleneck(pooled_x)
        return unpooled_x, pooled_x


class Decoder(torch.nn.Module):
    """Decoder part of the AutoEncoder."""
    
    def __init__(self,
                 nb_input_channels: int,
                 nb_channels_fst_feature_map: int,
                 latent_dim: int,
                 last_feature_size: int,
                 canny_edges: bool=False,
                 softmax: bool=False,
                 sigmoid: bool=False,
                 **kwargs) -> None:
        """Create an instance of a Decoder.

        Parameters
        ----------
        nb_input_channels : int
            Number of channels of the input data.
        nb_channels_fst_feature_map : int
            Number of channels of the first layer.
        latent_dim : int
            Dimension of the latent space.
        last_feature_size : tuple (int, int)
            length, width of the output of the first up-sampling layer
        canny_edges : bool, optional
            Whether data passed in encoder is Canny processed, if so last activation is a sigmoid, by default False.
        softmax : bool, optional
            Whether to use softmax as the activation function of the last layer, by default False.
        sigmoid : bool, optional
            Whether to use sigmoid as the activation function of the last layer, by default False.
        """

        super(Decoder, self).__init__()

        self.nb_input_channels = nb_input_channels
        self.nb_channels_fst_feature_map = nb_channels_fst_feature_map
        self.latent_dim = latent_dim
        self.last_feature_size = last_feature_size

        modules = []
        #  Set the number of channels per layer
        self.hidden_dims = [self.nb_channels_fst_feature_map, 
                            2*self.nb_channels_fst_feature_map, 4*self.nb_channels_fst_feature_map, 
                            8*self.nb_channels_fst_feature_map]
        
        # reverse since its up-sampling step
        self.hidden_dims.reverse()

        ## BUILD DECODER
        # first layer that maps embedding to 3D space
        self.linear_map = torch.nn.Linear(self.latent_dim, 
                                          self.last_feature_size[0] * self.last_feature_size[1] * 
                                          self.hidden_dims[0])
        
        # UP-SAMPLING
        for i in range(len(self.hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.hidden_dims[i], 
                                       out_channels=self.hidden_dims[i+1],
                                       kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(self.hidden_dims[i+1]),
                    nn.LeakyReLU())
            )
            
        if canny_edges or sigmoid:
            self.last_activation = nn.Sigmoid()
        elif softmax:
            self.last_activation = nn.Softmax()
        else:
            self.last_activation = nn.Identity()
            
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.hidden_dims[-1], 
                                   out_channels=self.nb_input_channels,
                                       kernel_size=2, stride=2, padding=0),
                self.last_activation
            )
        )
        
        self.decoder = nn.Sequential(*modules)
        
                
    def forward(self, x):

        z = self.linear_map(x)
        z = z.reshape((z.shape[0], self.hidden_dims[0], 
                       self.last_feature_size[0], self.last_feature_size[1]))
        z = self.decoder(z)
        return z
