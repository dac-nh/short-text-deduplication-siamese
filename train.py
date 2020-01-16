import argparse
import time
import warnings

import numpy as np
import torch
from torch import optim
from tqdm.notebook import tqdm

from model.model import TripletSiameseModel, TripletDistance
from utils.data_loader import (
    load_data_set,
    load_triplet_orders,
    load_padded_data,
    load_triplet,
    generate_embedding,
)

warnings.filterwarnings("ignore")

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


def prepare_data(file_path):
    """

    @param file_path: file to your pre-processed csv
    @return: df: Dataframe after processed
    @return: X: full-data as int64 matrix
    @return: X_len: original length of data
    @return: embeddings: character index embeddings index
    """
    # Load dataset
    df = load_data_set(file_path)
    df.fillna("", inplace=True)
    df.reset_index(inplace=True, drop=True)
    # Embedding
    embeddings = generate_embedding(embedding_index, embedding_dim=50)
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
    """
    Transfer your dataloader into CPU or GPU
    @param loader: DataLoader
    @param device: torch.device
    @return: dataloader in specific device
    """
    return [load.to(device) for load in loader]


def main(data_path,
         model_path,
         batch_size,
         epochs,
         early_stopping_steps,
         device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
         ):
    """

    @param data_path:
    @param model_path:
    @param batch_size:
    @param epochs:
    @param early_stopping_steps:
    @param device:
    """
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
    # Load model & optimizer
    lr = 0.01
    margin = 0.4
    model = TripletSiameseModel(
        embedding_dim=[len(embedding_index), 50],
        layers=1,
        hid_dim=50,
        n_classes=30,
        bidirectional=True
    ).to(device)
    distance = TripletDistance(margin=margin).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Load pre-trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # ---- Train model
    best_lost = None
    loss_list = []
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
        forward_index = 0
        if best_lost is None or best_lost > avg_loss:
            best_lost = avg_loss
            forward_index = 0
            # Save checkpoint every time we get the better loss
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                model_path,
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
    main(args.data_path, args.model_path, args.batch_size, args.epochs, args.early_stopping_steps)
