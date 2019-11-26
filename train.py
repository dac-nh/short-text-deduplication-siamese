import warnings

warnings.filterwarnings("ignore")
import torch
import numpy as np
import pandas as pd
import os
import itertools
import time

from tqdm import tqdm_notebook as tqdm
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from model.model_new import TripletSiameseModel, TripletDistance
from model.SelfAttentionModel import StructuredSelfAttention
from utils.data_loader_new import (
    load_data_set,
    load_word_to_index,
    load_triplet_orders,
    load_padded_data,
    load_triplet,
    generate_embedding
)
from utils.pretrained_glove_embeddings import load_glove_embeddings
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score

full_generated_data_path = "new_generated_labeled_data.csv"
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load dataset
df = load_data_set(full_generated_data_path, retrain=False)
df.fillna("", inplace=True)
df = df[df["cid"] != 50]
df.reset_index(inplace=True)
# get word to index and embedding whole dataset
word_to_index = load_word_to_index(df, retrain=False)
embeddings = generate_embedding(word_to_index, embedding_dim=300)
X, X_len = load_padded_data(df, word_to_index, retrain=False)

def truncate_non_string(X, X_len):
    # Drop rows that have length of word vector = 0
    truncate_index = [i for i in range(0, len(X_len)) if X_len[i] <= 0]
    X, X_len = (
        np.delete(X, truncate_index, axis=0),
        np.delete(X_len, truncate_index, axis=0),
    )

    return X, X_len, truncate_index

X, X_len, truncate_index = truncate_non_string(X, X_len)
df.drop(index=truncate_index, inplace=True)
df.reset_index(inplace=True)