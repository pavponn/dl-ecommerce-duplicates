import matplotlib.pyplot as plt
import numpy as np


class CommonVisualizer(object):

    def __init__(self):
        pass

    def plt_f1_score_vs_threshold(self, thresholds, f1_scores, model_name_and_loss, filename):
        plt.plot(thresholds, f1_scores)
        self._default_plt_setting(
            title=f'F1-score vs cosine similarity threshold ({model_name_and_loss})',
            x_label='Cosine similarity threshold',
            y_label='Average F1-score',
            filename=filename
        )

    def plt_losses(self, train_losses, val_losses, model_name, loss_name, filename):
        t_l = len(train_losses)
        v_l = len(val_losses)
        plt.plot(np.arange(1, t_l + 1), train_losses, label='Training loss')
        plt.plot(np.arange(1, v_l + 1), val_losses, label='Validation loss')

        self._default_plt_setting(
            title=f'Loss vs epochs ({model_name})',
            x_label='Epochs',
            y_label=loss_name,
            filename=filename
        )

    def _default_plt_setting(self, title, x_label, y_label, filename):
        plt.legend(loc='best')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename, dpi=500)

