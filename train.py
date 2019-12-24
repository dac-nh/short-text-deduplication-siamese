import warnings

warnings.filterwarnings("ignore")
import torch
import numpy as np
import pandas as pd
import os
import itertools
import time
import argparse

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


def to_cuda(loader, device):
    return [load.to(device) for load in loader]


def main(data_path, model_path, batch_size, epochs, early_stopping_steps):
    # ---- Load data and convert it to triplet
    df, X, X_len, embeddings = prepare_data(data_path)
    # Create data loader with batch
    df_triplet_orders = load_triplet_orders(df)
    print("Loading triplet order successfully!")
    anc_loader, pos_loader, neg_loader = load_triplet(
        np.array(X), X_len, df_triplet_orders, batch_size=batch_size
    )
    print("Load triplet data successfully!")

    # ---- Load triplet siamese model and distance
    model, distance, optimizer = load_triplet_siamese_model(
        model_path, embedding_index, 50,
    )

    # ---- Train model
    best_lost = None
    loss_list = []
    average_list = []
    model.train()

    start_time = time.time()
    for epoch in tqdm(range(epochs), desc="Epoch", total=epochs):
        avg_loss = 0
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
            # Load model and measure the distance between anchor, positive and negative
            x, pos, neg = model(anc_x, pos_x, neg_x)
            loss, pos_sim, neg_sim = distance(x, pos, neg)
            # Append to batch list
            avg_loss += float(loss)
            avg_pos_sim += pos_sim.mean()
            avg_neg_sim += neg_sim.mean()
            # Update weights
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
        # Save model thought each checkpoint
        # Early stopping after reachs {early_stopping_steps} steps
        if best_lost is None or best_lost > avg_loss:
            best_lost = avg_loss
            forward_index = 0
            # Save checkpoint every time we get the better loss
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                triplet_model_path,
            )
        else:
            forward_index += 1
            if forward_index == early_stopping_steps or best_lost == 0:
                break
    print("--- %s seconds ---" % (time.time() - start_time))


# ----- Main Function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/trained_data.csv",
        help="Path to folders of pre-processed data.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/trained_model.h5",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Number of triplet samples per batch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs is used to train model with new data.",
    )

    parser.add_argument(
        "--early_stopping_steps",
        type=int,
        default=5,
        help="Number of epochs is used to train model with new data.",
    )
    
    args, unknown = parser.parse_known_args()
    print(args)
#     main(args.data_path, args.model_path, args.batch_size, args.epochs, args.early_stopping_steps)