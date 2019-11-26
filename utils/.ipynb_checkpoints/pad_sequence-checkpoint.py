import torch
import numpy as np
 
class PadSequence:
    def __call__(self, batch):
        X, _ = zip(*batch)
        lengths = [len(x) for x in X]
        X_pad = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
        return X_pad, lengths