from math import ceil

import matplotlib.pyplot as plt


def plot_reconstructions_single_layer(imgs,
                                      layer_name,
                                      filters,
                                      n_cols=3,
                                      cell_size=4,
                                      save_fig=False):
    n_rows = ceil((len(imgs)) / n_cols)
    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(cell_size * n_cols, cell_size * n_rows))

    for i, ax in enumerate(axes.flat):
        ax.grid(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(f'fmap {filters[i]}')
        ax.imshow(imgs[i])

    fig.suptitle(f'{layer_name}', fontsize="x-large", y=1.0)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    save_name = layer_name.lower().replace(' ', '_')

    if save_fig:
        plt.savefig(
            f'{save_name}_fmaps_{"_".join([str(f) for f in filters])}.png')

    plt.show()

    return plt


def reconstructions_single_layer(FV,
                                 layer,
                                 filters,
                                 opt_steps=20,
                                 blur=5,
                                 lr=1e-1,
                                 print_losses=False):
    return [
        FV.visualize(layer=layer,
                     conv_filter=filters[i],
                     opt_steps=opt_steps,
                     blur=blur,
                     lr=lr,
                     print_losses=print_losses) for i in range(len(filters))]
