import yaml

from caevl.ae.autoencoder.ae_model import AutoEncoder


def load_ae(config):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    image_size = tuple(config['image_size'])

    nb_input_channels = config['AutoEncoder']['nb_input_channels']
    nb_channels_fst_feature_map = config['AutoEncoder']['nb_channels_fst_feature_map']
    latent_dim = config['AutoEncoder']['latent_dim']
    learning_rate = float(config['AutoEncoder']['learning_rate'])
    scheduler = config['AutoEncoder']['scheduler']
    loss_name = config['AutoEncoder']['loss_name']
    device = config['AutoEncoder']['device']
    alpha = config['AutoEncoder']['alpha']
    flag_perceptual_loss = config['AutoEncoder']['flag_perceptual_loss']
    canny_edges = config['AutoEncoder'].get('canny_edges', False)
    scalar_divide_channels_fst_feature_map_for_decoder = config['AutoEncoder'].get('scalar_divide_channels_fst_feature_map_for_decoder', 1)
    encoder_only = config['AutoEncoder'].get('encoder_only', False)
    softmax = config['AutoEncoder'].get('softmax', False)
    sigmoid = config['AutoEncoder'].get('sigmoid', False)
    use_cbam = config['AutoEncoder'].get('use_cbam', False)

    print(f'Using loss: {loss_name}. Using perceptual loss: {flag_perceptual_loss}')
    print(f'Using Canny edges: {canny_edges}.')

    autoencoder = AutoEncoder(input_dimensions=image_size,
                              nb_input_channels=nb_input_channels,
                              nb_channels_fst_feature_map=nb_channels_fst_feature_map,
                              latent_dim=latent_dim,
                              learning_rate=learning_rate,
                              scheduler=scheduler,
                              loss_name=loss_name,
                              device=device,
                              alpha=alpha,
                              flag_perceptual_loss=flag_perceptual_loss,
                              canny_edges=canny_edges,
                              scalar_divide_channels_fst_feature_map_for_decoder=scalar_divide_channels_fst_feature_map_for_decoder,
                              encoder_only=encoder_only,
                              softmax=softmax,
                              sigmoid=sigmoid,
                              use_cbam=use_cbam)

    return autoencoder
