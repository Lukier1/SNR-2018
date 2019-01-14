# MIT License
#
# Copyright (c) 2018 Bjarte Mehus Sunde, 2019 Jan Kumor
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a
    given patience."""
    def __init__(self, patience=7, verbose=False, f=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss
                improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.f = f
        self.checkpoint_saved = False

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} '
                  f'--> {val_loss:.6f}).  Saving model checkpoint...')
        torch.save(model.state_dict(), self._get_f())
        self.val_loss_min = val_loss
        self.checkpoint_saved = True

    def load_checkpoint(self, model):
        """Loads checkpoint into model."""
        if self.checkpoint_saved:
            model.load_state_dict(torch.load(self._get_f()))
        else:
            raise IOError("No checkpoint saved yet.")

    def _get_f(self):
        return 'checkpoint.pt' if self.f is None else self.f
