# %% md

# Load library and dataset

# %%

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
    load_char_to_index,
    load_triplet_orders,
    load_padded_data,
    load_triplet,
    generate_embedding,
)
from utils.pretrained_glove_embeddings import load_glove_embeddings
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# %%

main_path = "/data/dac/dedupe-project/openmap/"  # Open map dataset 


# main_path = "/data/dac/dedupe-project/new/"  # Lab dataset


# %%

def prepare_data(file_path, retrain=True):
    # Load dataset
    df = load_data_set(file_path, retrain=retrain)
    df.fillna("", inplace=True)
    df.reset_index(inplace=True, drop=True)
    # get char to index and embedding whole dataset
    embedding_index = load_char_to_index(df, retrain=retrain)
    embeddings = generate_embedding(embedding_index, embedding_dim=50)
    X, X_len = load_padded_data(df, embedding_index, char_level=True, retrain=retrain)

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

    return df, X, X_len, embeddings, embedding_index


def to_cuda(loader, device):
    return [load.to(device) for load in loader]


# %%

# set cuda device
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)

# random_augment_train.csv
# "new_generated_labeled_data.csv"
full_generated_data_path = "random_augment_train.csv"  # Local dataset
open_map_data_path = "openmap-us-train.csv"  # Open map dataset
# Remember to set open map in train data
df, X, X_len, embeddings, embedding_index = prepare_data(
    open_map_data_path, retrain=True
)

# %%

import matplotlib.pyplot as plt

plt.hist(X_len)
plt.savefig("/data/dac/dedupe-project/image/length_X.eps", format="eps", dpi=1200)

# %% md

# Building Model

# %% md

## Generate DataLoader and Triplet orders

# %%

batch_size = 700
df_triplet_orders = load_triplet_orders(df, retrain=True)
print("Loading triplet order successfully!")
anc_loader, pos_loader, neg_loader = load_triplet(
    np.array(X), X_len, df_triplet_orders, batch_size=batch_size, retrain=True
)
print("Load triplet data successfully!")

# %%

# Create train file as pair for some methods
train_pair_arr = []
for row in df_triplet_orders.loc[:20000, :].itertuples():
    anchor = row[2]
    pos = row[3]
    neg = row[4]

    pair = {}
    pair["address"] = df.iloc[anchor].content
    pair["duplicated_address"] = df.iloc[pos].content
    pair["similar"] = 1
    train_pair_arr.append(pair)

    pair = {}
    pair["address"] = df.iloc[anchor].content
    pair["duplicated_address"] = df.iloc[neg].content
    pair["similar"] = 0
    train_pair_arr.append(pair)

pd.DataFrame(train_pair_arr).to_csv(
    "/data/dac/dedupe-project/train_as_pair.csv", encoding="utf-8"
)

# %% md

## Triplet Siamese

# %%

# Self-attention triplet model
# triplet_siamese_300d_bi_gru: hit
# triplet_siamese_50d_bi_gru_random: outperform
triplet_model_path = (
    "/data/dac/dedupe-project/new/model/triplet_siamese_50d_bi_gru_random_openmap"
)

lr = 0.1
margin = 0.4
# Load model & optimizer
model = TripletSiameseModel(
    embeddings=embeddings, layers=1, hid_dim=50, n_classes=30
).to(device)
# model = StructuredSelfAttention(embeddings=embeddings, n_classes=50).to(device)
distance = TripletDistance(margin=margin).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
print(model, distance)

# %%

# Load model and optimizer
checkpoint = torch.load(triplet_model_path, map_location=device)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])

model.eval()

# %%

# Train model
epochs = 2
best_lost = None
early_stopping_steps = 5

loss_list = []
average_list = []
pos_sim_list = []  # Positive distance of all eposchs
neg_sim_list = []  # Negative distance of all epochs
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
        # Training model per batch
        # Send data to graphic card - Cuda
        anc_x, pos_x, neg_x = (
            to_cuda(anc_x, device),
            to_cuda(pos_x, device),
            to_cuda(neg_x, device),
        )
        x, pos, neg = model(anc_x, pos_x, neg_x)
        loss, pos_sim, neg_sim = distance(x, pos, neg)

        # Append to batch list
        avg_loss += float(loss)
        avg_pos_sim += pos_sim.mean()
        avg_neg_sim += neg_sim.mean()

        # F1 and Acc
        y_true = np.concatenate([np.ones(len(pos_sim)), np.zeros(len(pos_sim))])
        y_pred = np.concatenate(
            [
                [1 if y > 0.665 else 0 for y in pos_sim.to("cpu")],
                [1 if y > 0.665 else 0 for y in neg_sim.to("cpu")],
            ]
        )
        # Gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #         torch.cuda.empty_cache()  # Empty cuda cache
        print(
            "\rBatch:\t{}\tLoss:\t{}\t\tPos_sim:\t{}\t\tNeg_sim:\t{}\t\t".format(
                batch,
                round(float(loss), 4),
                round(float(pos_sim.mean()), 4),
                round(float(neg_sim.mean()), 4),
            ),
            end="",
        )
        print(
            "\t Accuracy:\t{}\t\tF1-score:\t{}\t\t".format(
                round(accuracy_score(y_true, y_pred), 4),
                round(f1_score(y_true, y_pred), 4),
            ),
            end="",
        )
    # Average loss and distance of all epochs
    avg_loss /= len(anc_loader)
    avg_pos_sim /= len(anc_loader)
    avg_neg_sim /= len(anc_loader)

    loss_list.append(avg_loss)
    pos_sim_list.append(avg_pos_sim)
    neg_sim_list.append(avg_neg_sim)

    print(
        "\rEpoch:\t{}\tAverage Loss:\t{}\t\tPos:\t{}\t\tNeg:\t{}\t\t".format(
            epoch,
            round(avg_loss, 4),
            round(float(avg_pos_sim), 4),
            round(float(avg_neg_sim), 4),
        )
    )
    if best_lost is None or best_lost > avg_loss:
        best_lost = avg_loss
        forward_index = 0
        #         Save model
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            triplet_model_path,
        )
    else:
        # Early stopping after reachs {early_stopping_steps} steps
        forward_index += 1
        if forward_index == early_stopping_steps or best_lost == 0:
            break
    if best_lost < 1:
        break
print("--- %s seconds ---" % (time.time() - start_time))

# %%

# Save model
torch.save(
    {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    triplet_model_path,
)

# %% md

# Test

# %%

from sklearn.metrics.pairwise import cosine_similarity

# %%

path = "/data/dac/dedupe-project/test/"
test_df = pd.read_csv(path + "test_address_3.csv", encoding="ISO-8859-1")
test_df.fillna("", inplace=True)
test_df.reset_index(inplace=True)
test_df_1 = test_df.loc[:, ["address"]]
test_df_1["content"] = (
    test_df_1["address"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
        .str.replace("null", "")
        .str.replace("nan", "")
)
test_df_2 = test_df.loc[:, ["duplicated_address"]]
test_df_2["content"] = (
    test_df_2["duplicated_address"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
        .str.replace("null", "")
        .str.replace("nan", "")
)


# %%

def data_loader(test_df_1, test_df_2):
    # Make data loader
    X1, X1_lens = load_padded_data(
        pd.DataFrame(test_df_1), embedding_index, dump_path=None, retrain=True
    )

    X2, X2_lens = load_padded_data(
        pd.DataFrame(test_df_2), embedding_index, dump_path=None, retrain=True
    )

    # Drop rows that have length of word vector = 0
    truncate_index = [
        i for i in range(0, len(X1_lens)) if (X1_lens[i] <= 0 or X2_lens[i] <= 0)
    ]
    X1, X1_lens = (
        np.delete(X1, truncate_index, axis=0),
        np.delete(X1_lens, truncate_index, axis=0),
    )
    X2, X2_lens = (
        np.delete(X2, truncate_index, axis=0),
        np.delete(X2_lens, truncate_index, axis=0),
    )

    def create_data_loader(X, batch_size=batch_size):
        X, X_lens = np.array(X[0]), np.array(X[1])

        # Create data loader
        data = TensorDataset(
            torch.from_numpy(X).type(torch.LongTensor), torch.ByteTensor(X_lens)
        )
        loader = DataLoader(data, batch_size=batch_size, drop_last=False)
        return loader

    return (
        create_data_loader([X1, X1_lens]),
        create_data_loader([X2, X2_lens]),
        truncate_index,
    )


def create_test(n, test_df_1, test_df_2):
    # Generate small test based on ground truth
    test_df_1a = pd.DataFrame()
    test_df_1b = pd.DataFrame()

    for i1, i2 in shuffle(list(itertools.combinations(test_df_1.index, 2)))[:n]:
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


# %% md

## True Test

# %%

X1, X2, truncate = data_loader(test_df_2, test_df_1)
test_df_1.drop(truncate, inplace=True)
test_df_1.reset_index(inplace=True, drop=True)
test_df_2.drop(truncate, inplace=True)
test_df_2.reset_index(inplace=True, drop=True)

pred_list = np.array([])
y_true = np.array([])
y_pred = np.array([])
att1_list = []
att2_list = []
start_time = time.time()
for a, b in tqdm(zip(X1, X2)):
    # Send data to graphic card - Cuda0
    a, b = to_cuda(a, device), to_cuda(b, device)
    with torch.no_grad():
        a, b = model(a, b)
        a, b = a.cpu(), b.cpu()
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
        #         att1 = att1.cpu()
        #         att2 = att2.cpu()
        dist = np.array(
            [
                cosine_similarity([a[i].numpy()], [b[i].numpy()])
                for i in range(0, len(a))
            ]
        ).flatten()

        y_true_curr = np.ones(len(dist))
        y_true = np.concatenate([y_true, y_true_curr])

        y_pred_curr = np.ones(len(dist))
        y_pred_curr[np.where(dist < 0.665)[0]] = 0
        y_pred = np.concatenate([y_pred, y_pred_curr])

        pred_list = np.concatenate([pred_list, dist])
#         att1_list.append(att1)
#         att2_list.append(att2)

print(
    "Accuracy:\t{}\t\tF1-score:\t{}\t\t".format(
        round(accuracy_score(y_true, y_pred), 4), round(f1_score(y_true, y_pred), 4)
    ),
    end="",
)
print(
    "Precision:\t{}\t\tRecall:\t{}\t\t".format(
        round(precision_score(y_true, y_pred), 4),
        round(recall_score(y_true, y_pred), 4),
    ),
    end="",
)
print(time.time() - start_time)


# %%

def test(n, df1, df2):
    total_acc = 0
    total_f1 = 0
    for i in range(0, 1):
        test_df_1a, test_df_1b = create_test(n, df1, df2)
        X1, X2, drop = data_loader(test_df_1a, test_df_1b)

        pred_list = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        # att1_list = []
        # att2_list = []
        start_time = time.time()

        for a, b in tqdm(zip(X1, X2)):
            # Send data to graphic card - Cuda0
            a, b = to_cuda(a, device), to_cuda(b, device)
            with torch.no_grad():
                a, b = model(a, b)
                a, b = a.cpu(), b.cpu()
                a = a.reshape(a.shape[0], -1)
                b = b.reshape(b.shape[0], -1)
                #         att1 = att1.cpu()
                #         att2 = att2.cpu()
                dist = np.array(
                    [
                        cosine_similarity([a[i].numpy()], [b[i].numpy()])
                        for i in range(0, len(a))
                    ]
                ).flatten()

                y_true_curr = np.zeros(len(dist))
                y_true = np.concatenate([y_true, y_true_curr])
                if len(y_true) >= n:
                    y_true[n:] = 1

                y_pred_curr = np.ones(len(dist))
                y_pred_curr[np.where(dist <= 0.72)[0]] = 0
                y_pred = np.concatenate([y_pred, y_pred_curr])

                pred_list = np.concatenate([pred_list, dist])
        #         att1_list.append(att1)
        #         att2_list.append(att2)
        total_acc += accuracy_score(y_true, y_pred)
        total_f1 += f1_score(y_true, y_pred)
        print(
            "Accuracy:\t{}\t\tF1-score:\t{}\t\t".format(
                round(accuracy_score(y_true, y_pred), 4),
                round(f1_score(y_true, y_pred), 4),
            ),
            end="",
        )
        print(
            "Precision:\t{}\t\tRecall:\t{}\t\t".format(
                round(precision_score(y_true, y_pred), 4),
                round(recall_score(y_true, y_pred), 4),
            ),
            end="",
        )

    print(
        "\nAccuracy:\t{}\t\tF1-score:\t{}\t\t".format(
            round(total_acc, 4), round(total_f1, 4)
        ),
        end="",
    )
    print(time.time() - start_time)
    return test_df_1a, test_df_1b


# %% md

## Test 1

# %%

test_df_1a, test_df_1b = test(1176, test_df_1, test_df_2)

# %% md

## Test 2

# %%

test_df_new = pd.read_csv(path + "test.csv")
# drop_single = [i for i in test_df_new.ID.unique() if sum(test_df_new.ID == i) <2]
# test_df_new.set_index('ID', inplace=True)
# test_df_new.drop(drop_single, inplace=True)

# # Generate pairs
# pairs = []
# for i in test_df_new.index.unique():
#     temp = test_df_new[test_df_new.index == i]
#     # Generate a pair on the same row and append them to pairs
#     pair = [temp.iloc[0, 1], temp.iloc[1, 1]]
#     pairs.append(pair)

# test_df_new = pd.DataFrame(pairs, columns=['address', 'duplicated_address'])

test_df_new.fillna("", inplace=True)
test_df_new.reset_index(inplace=True)
test_df_1 = test_df_new.loc[:, ["address"]]
test_df_1["content"] = (
    test_df_1["address"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
        .str.replace("null", "")
        .str.replace("nan", "")
)
test_df_2 = test_df_new.loc[:, ["duplicated_address"]]
test_df_2["content"] = (
    test_df_2["duplicated_address"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
        .str.replace("null", "")
        .str.replace("nan", "")
)


# %%

def test(n, df1, df2):
    total_acc = 0
    total_f1 = 0
    for i in range(0, 1):
        test_df_1a, test_df_1b = create_test(n, df1, df2)
        X1, X2, drop = data_loader(test_df_1a, test_df_1b)

        pred_list = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        # att1_list = []
        # att2_list = []
        for a, b in tqdm(zip(X1, X2)):
            # Send data to graphic card - Cuda0
            a, b = to_cuda(a, device), to_cuda(b, device)
            with torch.no_grad():
                a, b = model(a, b)
                a, b = a.cpu(), b.cpu()
                a = a.reshape(a.shape[0], -1)
                b = b.reshape(b.shape[0], -1)
                #         att1 = att1.cpu()
                #         att2 = att2.cpu()
                dist = np.array(
                    [
                        cosine_similarity([a[i].numpy()], [b[i].numpy()])
                        for i in range(0, len(a))
                    ]
                ).flatten()

                y_true_curr = np.zeros(len(dist))
                y_true = np.concatenate([y_true, y_true_curr])
                if len(y_true) >= n:
                    y_true[n:] = 1

                y_pred_curr = np.ones(len(dist))
                y_pred_curr[np.where(dist <= 0.72)[0]] = 0
                y_pred = np.concatenate([y_pred, y_pred_curr])

                pred_list = np.concatenate([pred_list, dist])
        #         att1_list.append(att1)
        #         att2_list.append(att2)
        total_acc += accuracy_score(y_true, y_pred)
        total_f1 += f1_score(y_true, y_pred)
        print(
            "Accuracy:\t{}\t\tF1-score:\t{}\t\t".format(
                round(accuracy_score(y_true, y_pred), 4),
                round(f1_score(y_true, y_pred), 4),
            ),
            end="",
        )
        print(
            "Precision:\t{}\t\tRecall:\t{}\t\t".format(
                round(precision_score(y_true, y_pred), 4),
                round(recall_score(y_true, y_pred), 4),
            ),
            end="",
        )

    print(
        "\nAccuracy:\t{}\t\tF1-score:\t{}\t\t".format(
            round(total_acc, 4), round(total_f1, 4)
        ),
        end="",
    )
    return test_df_1a, test_df_1b


test_df_1a, test_df_1b = test(5000, test_df_1, test_df_2)

# %% md

## Mistake

# %%

test_df = pd.read_csv(
    "/data/dac/dedupe-project/test/new/GT_added.csv", encoding="ISO-8859-1"
)
mistakes_df = pd.read_excel("/data/dac/dedupe-project/test/new/mistakes.xlsx")

# %%

mistakes_df.columns

# %%

mistakes = [
    "misunderstanding",
    "typing error (translate)",
    "don’t know zipcode",
]
mistake_dict = {
    mistake: list(mistakes_df[mistake].dropna().index) for mistake in mistakes
}

# %%

test_df.fillna("", inplace=True)
test_df.reset_index(inplace=True)
test_df_1 = test_df.loc[:, ["address"]]
test_df_1["content"] = (
    test_df_1["address"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
)
test_df_2 = test_df.loc[:, ["duplicated_address"]]
test_df_2["content"] = (
    test_df_2["duplicated_address"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
)

# %%

X1, X2, truncate = data_loader(test_df_2, test_df_1)
test_df_1.drop(truncate, inplace=True)
# test_df_1.reset_index(inplace=True)
test_df_2.drop(truncate, inplace=True)
# test_df_2.reset_index(inplace=True)

pred_list = np.array([])
y_true = np.array([])
y_pred = np.array([])
att1_list = []
att2_list = []
for a, b in tqdm(zip(X1, X2)):
    # Send data to graphic card - Cuda0
    a, b = to_cuda(a), to_cuda(b)
    with torch.no_grad():
        a, b = model(a, b)
        a, b = a.cpu(), b.cpu()
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
        #         att1 = att1.cpu()
        #         att2 = att2.cpu()
        dist = np.array(
            [
                cosine_similarity([a[i].numpy()], [b[i].numpy()])
                for i in range(0, len(a))
            ]
        ).flatten()

        y_true_curr = np.ones(len(dist))
        y_true = np.concatenate([y_true, y_true_curr])

        y_pred_curr = np.ones(len(dist))
        y_pred_curr[np.where(dist <= 0.83)[0]] = 0
        y_pred = np.concatenate([y_pred, y_pred_curr])

        pred_list = np.concatenate([pred_list, dist])
#         att1_list.append(att1)
#         att2_list.append(att2)

print(
    "Accuracy:\t{}\t\tF1-score:\t{}\t\t".format(
        round(accuracy_score(y_true, y_pred), 4), round(f1_score(y_true, y_pred), 4)
    ),
    end="",
)
print(
    "Precision:\t{}\t\tRecall:\t{}\t\t".format(
        round(precision_score(y_true, y_pred), 4),
        round(recall_score(y_true, y_pred), 4),
    ),
    end="",
)

# %%

for key, value in mistake_dict.items():
    y_t = y_true[value]
    y_p = y_pred[value]
    print(key)
    print(
        "Accuracy:\t{}\t\tF1-score:\t{}\t\t".format(
            round(accuracy_score(y_t, y_p), 4), round(f1_score(y_t, y_p), 4)
        ),
        end="",
    )
    print(
        "Precision:\t{}\t\tRecall:\t{}\t\t".format(
            round(precision_score(y_t, y_p), 4), round(recall_score(y_t, y_p), 4),
        )
    )

# %%

test_df_1.to_csv("/data/dac/dedupe-project/test/new/GT_added.csv", encoding="utf-8")
