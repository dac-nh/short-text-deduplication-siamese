import itertools
import os
import pickle
from multiprocessing.pool import ThreadPool

import nltk
import numpy as np
import pandas as pd
import torch
import spacy
from keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pack_padded_sequence  # Packed padded sequence
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook as tqdm

nltk.download("punkt")
main_path = "/data/dac/dedupe-project/"
nlp = spacy.load('en_core_web_lg')

def load_data_set(
    file_path="generated_labeled_data.csv", dump_path="prepro_data.csv", retrain=False
):
    """
    Loads the dataset with self-attraction embedding
    :param dump_path: save preprocessed data (None for not save)
    :param file_path: main file 
    :param retrain: if you want to retrain model
    :return: dataset
    """
    file_path = main_path + file_path
    if dump_path is not None and os.path.isfile(dump_path) and not retrain:
        dump_path = main_path + dump_path
        df = pd.read_csv(dump_path)
        return df

    df = pd.read_csv(file_path)
    # lowercase all word of name and address columns
    df["name"], df["address"] = df["name"].str.lower(), df["address"].str.lower()
    df.fillna("", inplace=True)
    df["content"] = df["name"] + " " + df["address"] + " " + df["postal"]
    # Simple pre-processing
    df["content"] = (
        df["content"]
        .str.replace("\n", " ")
        .str.replace(",", " ")
        .str.replace(r"[ ]+", " ", regex=True)
    )
    if dump_path is not None:
        dump_path = main_path + dump_path
        df.to_csv(dump_path, index=False)  # Save to new file
    return df


def load_word_to_index(df, dump_path="embedding/word_to_index.pkl", retrain=False):
    """
    Create word to index dictionary
    :param dump_path: path to save pre-run file
    :param retrain: bool: if file is exist and want to retrain model
    :param df: dataframe
    :return: word_to_index
    """
    dump_path = main_path + dump_path
    # tokenize and get word_to_index
    if os.path.isfile(dump_path) and not retrain:
        word_to_index = pickle.load(open(dump_path, "rb"))
    else:
        # generate new word_to_index if not exist
        content_all = df.loc[:, "content"].str.cat(sep=" ")
        words_tokenized = nltk.word_tokenize(content_all)
        word_to_index = {
            token: index + 3 for index, token in enumerate(set(words_tokenized))
        }
        word_to_index["<PAD>"] = 0
        word_to_index["<START"] = 1
        word_to_index["<UNK>"] = 2
        pickle.dump(word_to_index, open(dump_path, "wb"))

    return word_to_index


def generate_embedding(word_to_index, embedding_dim=50):
    # Generate Embedding dimensions
    return np.zeros([len(word2idx), embedding_dim])
    

def load_padded_data(
    df, word_to_index, dump_path="embedding/x_train_pad.pkl", pad_size=50, retrain=False
):
    if dump_path is not None:
        dump_path = main_path + dump_path
        if os.path.isfile(dump_path) and not retrain:
            x_train_pad = pickle.load(open(dump_path, "rb"))
            return x_train_pad

    # Load dataset as triplet samples
    x_train = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Padding"):
        content = row["content"]
        words_tokenized = nltk.word_tokenize(content)
        # Create word vector
        words_vector = []
        for word in words_tokenized:
            word_index = word_to_index.get(word)
            if word_index is None or word_index < 0:
                # If word not exists in word_to_index
                word_index = 0
            words_vector.append(word_index)
            # Add to whole dataset
        x_train.append(words_vector)
    x_train_pad = pad_sequences(x_train, maxlen=pad_size, padding="post")
    if dump_path is not None:
        pickle.dump(x_train_pad, open(dump_path, "wb"))

    return x_train_pad


def data_to_vec(
    df, word_to_index, dump_path="embedding/x_train_to_word_vec.pkl", retrain=False
):
    if dump_path is not None:
        dump_path = main_path + dump_path
        if os.path.isfile(dump_path) and not retrain:
            x_train = pickle.load(open(dump_path, "rb"))
            return x_train

    # Load dataset as triplet samples
    x_train = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Padding"):
        content = row["content"]
        words_tokenized = nltk.word_tokenize(content)
        # Create word vector
        words_vector = []
        for word in words_tokenized:
            word_index = word_to_index.get(word)
            if word_index is None or word_index < 0:
                # If word not exists in word_to_index
                word_index = 0
            words_vector.append(word_index)
            # Add to whole dataset
        x_train.append(words_vector)
    if dump_path is not None:
        pickle.dump(x_train, open(dump_path, "wb"))

    return x_train


def load_triplet_orders(df, dump_path="triplet_samples_id.csv", retrain=False):
    """
    Create triplet samples
    :param df:
    :param retrain:
    :param dump_path:
    :return: result with code and content (DataFrame)
    """
    dump_path = main_path + dump_path
    if os.path.isfile(dump_path) and not retrain:
        # Load file without train
        df_result = pd.read_csv(dump_path).iloc[:, 1:]
        result = {"code": 1, "content": df_result}
        return result

    # Generate triplet samples
    cid_list = list(df["cid"].unique())

    def generate_triplet_sample(cid):
        """
        Combine triplet samples
        :param cid:
        :return:
        """
        df_current_pos = df[(df["cid"] == cid) & (df["similar"] == 1)]
        df_current_neg = df[(df["cid"] == cid) & (df["similar"] == 0)]
        # Generate all possible positive
        similar_pairs = list(itertools.combinations(df_current_pos.index, 2))
        triplet_order_arr = []
        for each_pair_positive in similar_pairs:
            for neg_sample in df_current_neg.index.values:
                # Link each pair of positive to a negative
                triplet = (cid,) + each_pair_positive + (neg_sample,)
                triplet_order_arr.append(triplet)
        return pd.DataFrame(triplet_order_arr)

    # Retrain false
    threads = list()
    pool = ThreadPool()
    for cid in tqdm(cid_list, desc="[Generate Triplet Dataset] Start thread"):
        # Iter all clusters to create all triplet samples: 1 anchor, 1 positive and 1 negative
        thread = pool.apply_async(generate_triplet_sample, (cid,))
        threads.append(thread)
    # Join all triplet sample into one DataFrame
    df_result = pd.DataFrame()
    for thread in tqdm(threads, desc="[Generate Triplet Dataset] Join thread"):
        try:
            df_result = pd.concat(
                [df_result, thread.get()]
            )  # Append triplet samples of each cluster to DataFrame
        except:
            print("[generate_triplet_samples] Problem in joining Thread Pool")
    df_result.rename(columns={0: "cid", 1: "anchor", 2: "pos", 3: "neg"}, inplace=True)
    df_result = shuffle(df_result)
    df_result.reset_index(inplace=True)
    df_result.to_csv(dump_path, index=False)  # Save to file
    result = {"code": 1, "content": df_result}
    return result


def load_triplet(
    x_padded,
    df_triplet_orders,
    batch_size=520,
    dump_path="embedding/triplet_data.pkl",
    retrain=False,
    embedded=False,
):
    dump_path = main_path + dump_path
    if os.path.isfile(dump_path) and not retrain:
        anc_arr, pos_arr, neg_arr = pickle.load(open(dump_path, "rb"))
    else:
        #         df_triplet_orders = shuffle(df_triplet_orders)
        anc_arr, pos_arr, neg_arr = [], [], []
        for index, row in tqdm(
            df_triplet_orders.iloc[:, :].iterrows(),
            total=df_triplet_orders.shape[0],
            desc="Load triplets",
        ):
            anc_loc, pos_loc, neg_loc = row[["anchor", "pos", "neg"]]
            anc_arr.append(x_padded[anc_loc])
            pos_arr.append(x_padded[pos_loc])
            neg_arr.append(x_padded[neg_loc])

        pickle.dump([anc_arr, pos_arr, neg_arr], open(dump_path, "wb"))

    def create_data_loader(array, batch_size=batch_size):
        # Create data loader
        if embedded:
            # if data does not need to embedding inside model
            data = TensorDataset(torch.from_numpy(array).to(torch.FloatTensor))
        else:
            data = TensorDataset(torch.from_numpy(array).to(torch.LongTensor))
        loader = DataLoader(data, batch_size=batch_size, drop_last=False)
        return loader

    anc_loader = create_data_loader(np.array(anc_arr))
    pos_loader = create_data_loader(np.array(pos_arr))
    neg_loader = create_data_loader(np.array(neg_arr))

    return anc_loader, pos_loader, neg_loader