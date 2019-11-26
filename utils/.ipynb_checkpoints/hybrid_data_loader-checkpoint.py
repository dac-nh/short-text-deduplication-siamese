import itertools
import os
import pickle
from multiprocessing.pool import ThreadPool

import nltk
import numpy as np
import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook as tqdm