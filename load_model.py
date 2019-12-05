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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def paring(df_pred):
    df_1 = pd.DataFrame()
    df_2 = pd.DataFrame()
    # Create pair to find duplication
    pair_index = list(itertools.combinations(df_pred.index, 2))

    for i1, i2 in pair_index:
        try:
            test_df_1a = test_df_1a.append(test_df_1.iloc[i1, :])
            test_df_1b = test_df_1b.append(test_df_2.iloc[i2, :])
        except:
            print(i1, i2)

    test_df_1b = test_df_1b.append(test_df_1)
    test_df_1a = test_df_1a.append(test_df_2)

    test_df_1a.reset_index(inplace=True)
    test_df_1b.reset_index(inplace=True)

    return test_df_1a, test_df_1b


def load_model(df_pred, model_path, device):
    # ---- PREPARE ENVIRONMENT AND DATA TO LOAD MODEL
    # Load dataset
    print("Using", device, "to process!")
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

    # ---- LOAD MODEL
    batch_size = 10000
    triplet_model_path = path
    lr = 0.1
    margin = 0.4
    # Load model & optimizer
    model = TripletSiameseModel(
        embedding_dim=[len(embedding_index), 50],
        layers=1,
        hid_dim=50,
        max_len=5,
        n_classes=30,
    ).to(device)
    distance = TripletDistance(margin=margin).to(device)  # Distance metric
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Load optimizer
    print("Init model successfully!")

    # Load model and optimizer
    checkpoint = torch.load(triplet_model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    model.eval()
    print("Load model successfully!")

    # ---- PREDICT PART
    df_pred = df_pred.loc[:, ['ID', 'content_1', 'content_2']]
    df_pred.fillna("", inplace=True)
    df_pred_1 = df_pred.loc[:, ['ID', "content_1"]]
    df_pred_1["content"] = (
        df_pred_1["content_1"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
        .str.replace("null", "")
        .str.replace("nan", "")
    )
    
    df_pred_2 = df_pred.loc[:, ['ID', "content_2"]]
    df_pred_2["content"] = (
        df_pred_2["content_2"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
        .str.replace("null", "")
        .str.replace("nan", "")
    )