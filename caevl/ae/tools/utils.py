import matplotlib.pyplot as plt
import torch


def tensor_to_im(tensor):
    return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()


def plot_images(images,
                n_cols=8,
                n_rows=1,
                float_=True,
                title='',
                show=True):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    def plot_row(ax_row, images_row):
        for i, im in enumerate(images_row):
            if i >= n_cols:
                break
            ax_row[i].imshow(im, cmap='gray')
            ax_row[i].axis('off')

    if n_rows == 1:
        plot_row(axs, images)
    else:
        for n_row in range(n_rows):
            plot_row(axs[n_row], images[n_row])
    plt.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()


def plot_first_reconstruction(loader, model, nb_images_to_plot=8, title='', float_=True, scalar=1):
    images, _ = next(iter(loader))
    reconstructions, _ = model(images.to(model.device))

    n_channels = reconstructions.shape[1]
    if n_channels > 3:
        images = torch.argmax(images, dim=1, keepdims=True) / n_channels
        reconstructions = torch.argmax(reconstructions, dim=1, keepdims=True) / n_channels

    images_numpy = tensor_to_im(images)
    reconstructions_numpy = tensor_to_im(reconstructions)

    plot_images([images_numpy, reconstructions_numpy], n_cols=nb_images_to_plot, n_rows=2, title=title, float_=float_)
