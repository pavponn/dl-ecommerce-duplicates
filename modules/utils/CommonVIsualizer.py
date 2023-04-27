import matplotlib.pyplot as plt
import numpy as np


class CommonVisualizer(object):

    def __init__(self):
        pass

    def plt_f1_score_vs_threshold(self, thresholds, f1_scores, model_name_and_loss, filename):
        plt.plot(thresholds, f1_scores)
        self._default_plt_setting(
            title=f'F1-score vs cosine similarity threshold, ({model_name_and_loss})',
            x_label='Cosine similarity threshold',
            y_label='Average F1-score',
            filename='filename'
        )

    def _default_plt_setting(self, title, x_label, y_label, filename):
        plt.legend(loc='best')
        plt.xticks(rotation=30)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename)
