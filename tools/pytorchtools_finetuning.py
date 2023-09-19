'''
    Early stopping (fine-tuning version)
'''

import numpy as np
import torch

class EarlyStopping:

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model1, model2, model3, model4):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model1, model2, model3, model4)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model1, model2, model3, model4)
            self.counter = 0

    def checkpoint(self, val_loss, model1, model2, model3, model4):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
                        'background_extraction_module' : model1.state_dict(),
                        'text_extraction_module' : model2.state_dict(),
                        'selective_word_removal_module' : model3.state_dict(),
                        'reconstruction_module' : model4.state_dict()}, self.path)
        self.val_loss_min = val_loss