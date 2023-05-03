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
        plt.show()

        
    def plt_combined_f1_score(self, img_f1_scores, text_f1_scores, combined_f1_scores, model_name, filename):
        i_l = len(img_f1_scores)
        t_l = len(text_f1_scores)
        c_l = len(combined_f1_scores)
        plt.plot(np.arange(1, i_l + 1), img_f1_scores, label='Image Only')
        plt.plot(np.arange(1, t_l + 1), text_f1_scores, label='Text Only')
        plt.plot(np.arange(1, c_l + 1), combined_f1_scores, label='Combined')

        self._default_plt_setting(
            title=f'Loss vs epochs ({model_name})',
            x_label='Epochs',
            y_label=loss_name,
            filename=filename
        )
        


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
        plt.show()

        
    def plt_combined_f1_score(
        self, 
        valid_img_f1_scores, 
        valid_text_f1_scores, 
        valid_combined_f1_scores, 
        train_img_f1_scores, 
        train_text_f1_scores, 
        train_combined_f1_scores, 
        full_img_f1_scores, 
        full_text_f1_scores, 
        full_combined_f1_scores, 
        model_name, 
        filename):
        thl = len(valid_img_f1_scores)
        plt.plot(np.arange(1, thl + 1), valid_img_f1_scores, label='Image Only - Valid')
        plt.plot(np.arange(1, thl + 1), valid_text_f1_scores, label='Text Only - Valid')
        plt.plot(np.arange(1, thl + 1), valid_combined_f1_scores, label='Combined - Valid')
        plt.plot(np.arange(1, thl + 1), train_img_f1_scores, label='Image Only - Train')
        plt.plot(np.arange(1, thl + 1), train_text_f1_scores, label='Text Only - Train')
        plt.plot(np.arange(1, thl + 1), train_combined_f1_scores, label='Combined - Train')
        plt.plot(np.arange(1, thl + 1), full_img_f1_scores, label='Image Only - Full')
        plt.plot(np.arange(1, thl + 1), full_text_f1_scores, label='Text Only - Full')
        plt.plot(np.arange(1, thl + 1), full_combined_f1_scores, label='Combined - Full')

        self._default_plt_setting(
            title=f'F1-score vs cosine similarity threshold ({model_name})',
            x_label='Cosine similarity threshold',
            y_label='Average F1-score',
            filename=filename
        )

