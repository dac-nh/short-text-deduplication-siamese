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
from model.model import TripletSiameseModel, TripletDistance
from model.SelfAttentionModel import StructuredSelfAttention
from utils.data_loader import (
    load_data_set,
    load_word_to_index,
    load_char_to_index,
    load_triplet_orders,
    load_padded_data,
    load_triplet,
    generate_embedding,
)
from utils.pretrained_glove_embeddings import load_glove_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


# Load embedding
embedding_index = {
    " ": 1,
    "!": 2,
    "#": 3,
    "$": 4,
    "%": 5,
    "&": 6,
    "'": 7,
    "(": 8,
    ")": 9,
    "*": 10,
    ",": 11,
    "-": 12,
    ".": 13,
    "/": 14,
    "0": 15,
    "1": 16,
    "2": 17,
    "3": 18,
    "4": 19,
    "5": 20,
    "6": 21,
    "7": 22,
    "8": 23,
    "9": 24,
    ":": 25,
    ";": 26,
    "<": 27,
    ">": 28,
    "?": 29,
    "@": 30,
    "^": 31,
    "a": 32,
    "b": 33,
    "c": 34,
    "d": 35,
    "e": 36,
    "f": 37,
    "g": 38,
    "h": 39,
    "i": 40,
    "j": 41,
    "k": 42,
    "l": 43,
    "m": 44,
    "n": 45,
    "o": 46,
    "p": 47,
    "q": 48,
    "r": 49,
    "s": 50,
    "t": 51,
    "u": 52,
    "v": 53,
    "w": 54,
    "x": 55,
    "y": 56,
    "z": 57,
    "，": 58,
    "<PAD>": 0,
}


def prepare_data(file_path, retrain=True):
    # Load dataset
    df = load_data_set(file_path)
    df.fillna("", inplace=True)
    df.reset_index(inplace=True, drop=True)
    # Embedding
    embeddings = generate_embedding(embedding_index, embedding_dim=embedding_dim)
    # Get X and X_len as matrix
    X, X_len = load_padded_data(df, embedding_index, char_level=True)

    def truncate_non_string(X, X_len):
        # Drop rows that have length of word vector = 0
        truncate_index = [i for i in range(0, len(X_len)) if X_len[i] <= 0]
        X, X_len = (
            np.delete(X, truncate_index, axis=0),
            np.delete(X_len, truncate_index, axis=0),
        )

        return X, X_len, sorted(truncate_index, reverse=True)

    X, X_len, truncate_index = truncate_non_string(X, X_len)
    df.drop(index=truncate_index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df, X, X_len, embeddings


# ---- Load data and convert it to triplet
df, X, X_len, embeddings = prepare_data(main_path + full_generated_data_path)
# Create data loader with batch
batch_size = 10000
df_triplet_orders = load_triplet_orders(df)
print("Loading triplet order successfully!")
anc_loader, pos_loader, neg_loader = load_triplet(
    np.array(X), X_len, df_triplet_orders, batch_size=batch_size
)
print("Load triplet data successfully!")


# ---- Load triplet siamese model and distance
model, distance, optimizer = load_triplet_siamese_model("/data/dac/dedupe-project/new/model/triplet_siamese_50d_bi_gru_random", embedding_index, 50)

# ---- Train model
epochs = 7
best_lost = None
early_stopping_steps = 5
loss_list = []
average_list = []
model.train()

start_time = time.time()
for epoch in tqdm(range(epochs), desc="Epoch", total=epochs):
    avg_loss = 0
    avg_acc = 0
    avg_pos_sim = 0
    avg_neg_sim = 0
    for batch, [anc_x, pos_x, neg_x] in enumerate(
        zip(anc_loader, pos_loader, neg_loader)
    ):
        # Send data to graphic card - Cuda
        anc_x, pos_x, neg_x = (
            to_cuda(anc_x, device),
            to_cuda(pos_x, device),
            to_cuda(neg_x, device),
        )
        # Load model and compute the distance
        x, pos, neg = model(anc_x, pos_x, neg_x)
        loss, pos_sim, neg_sim = distance(x, pos, neg)

        # Append to batch list
        avg_loss += float(loss)
        avg_pos_sim += pos_sim.mean()
        avg_neg_sim += neg_sim.mean()

        # Gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Average loss and distance of all epochs
    avg_loss /= len(anc_loader)
    avg_pos_sim /= len(anc_loader)
    avg_neg_sim /= len(anc_loader)

    loss_list.append(avg_loss)
    print(
        "\rEpoch:\t{}\tAverage Loss:\t{}\t\tPos:\t{}\t\tNeg:\t{}\t\t".format(
            epoch,
            round(avg_loss, 4),
            round(float(avg_pos_sim), 4),
            round(float(avg_neg_sim), 4),
        ),
        end="",
    )
    if best_lost is None or best_lost > avg_loss:
        best_lost = avg_loss
        forward_index = 0
        #Save model
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            triplet_model_path,
        )
    else:
        # Early stopping after reachs {early_stopping_steps} steps
        forward_index += 1
        if forward_index == early_stopping_steps or best_lost == 0:
            break
print("--- %s seconds ---" % (time.time() - start_time))