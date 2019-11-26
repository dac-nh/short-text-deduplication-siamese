import itertools
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.utils import shuffle
# from keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook as tqdm

# main_path = '/'  # Use at local
nlp = spacy.load("en_core_web_lg")
nlp.max_length = 4500000  # Increase max length


def load_data_set(
        file_path="generated_labeled_data.csv", retrain=False
):
    """
    Loads the dataset with self-attraction embedding
    :param file_path: main file 
    :param retrain: if you want to retrain model
    :return: dataframe
    """

    df = pd.read_csv(file_path)
    df["content"] = (
        df["address"]
            .str.lower()
            .str.replace("\n", " ")
            .str.replace(r"[ ]+", " ", regex=True)
            .str.replace("null", "")
            .str.replace("nan", "")
    )
    return df


def load_word_to_index(df, retrain=False):
    """
    Create word to index dictionary
    :param retrain: bool: if file is exist and want to retrain model
    :param df: dataframe
    :return: word_to_index as a dictionary
    """
    # generate new word_to_index if not exist
    content_all = df.loc[:, "content"].str.cat(sep=" ")
    words_tokenized = [
        word.text for word in nlp(content_all) if word.pos_ not in ["PUNCT", "SYM"]
    ]  # Remove all words are punct and sym
    word_to_index = {
        token: index + 1 for index, token in enumerate(set(words_tokenized))
    }
    word_to_index["<PAD>"] = 0
    print("Load word to index successfully!")
    return word_to_index


def load_char_to_index(df, retrain=False):
    """
    Create character to index dictionary
    :param retrain: bool: if file is exist and want to retrain model
    :param df: dataframe
    :return: char_to_index as a dictionary
    """
    # generate new char_to_index if not exist
    content_all = df.loc[:, "content"].str.cat(sep=" ")
    char_to_index = {
        char: key + 1 for key, char in enumerate(sorted(list(set(content_all))))
    }
    char_to_index["<PAD>"] = 0
    print("Load character to index successfully!")
    return char_to_index


def generate_embedding(word_to_index, embedding_dim=50):
    # Generate Embedding dimensions
    embeddings = np.zeros([len(word_to_index), embedding_dim])
    return torch.from_numpy(np.array(embeddings)).float()


def truncate_non_string(X, X_len):
    # Drop rows that have length of word vector = 0
    truncate_index = [i for i in range(0, len(X_len)) if X_len[i] <= 0]
    X, X_len = (
        np.delete(X, truncate_index, axis=0),
        np.delete(X_len, truncate_index, axis=0),
    )

    return X, X_len, truncate_index


def load_padded_data(
        df,
        word_to_index,
        char_level=True,
        retrain=False,
):
    """
    Padding data into a fixed length
    :param df: dataframe
    :param word_to_index: dictionary
    :param retrain: (default is False) True if you want to retrain
    :return: x_train_pad: padded data
    :return: x_train_length: original length of all data (in case you want to unpadded)
    """
    x_train = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Padding"):

        # Create word vector
        if char_level:
            words_vector = [word_to_index.get(word) for word in row["content"] if word_to_index.get(word) is not None]
        else:
            # content to word vector though each word
            words_tokenized = nlp(row["content"])
            words_vector = [
                word_to_index.get(word.text)
                for word in words_tokenized
                if (
                        word_to_index.get(word.text) is not None
                        and word.pos_ not in ["PUNCT", "SYM"]
                )
            ]
        x_train.append(torch.LongTensor(words_vector))

    x_train_len = [
        len(x) for x in x_train
    ]  # Get length for pack_padded_sequence after to remove padding
    x_train_pad = pad_sequence(x_train, batch_first=True)
    print("Load padded data successfully!")
    return x_train_pad, x_train_len


def load_triplet_orders(df, retrain=False):
    """
    Create triplet samples
    :param df:  dataframe
    :param retrain:  True if you want to re-generate triplet order (Default False)
    :return: result with code and content (DataFrame)
    """
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

        if len(similar_pairs) < 2:
            return None

        triplet_order_arr = []
        for each_pair_positive in similar_pairs:
            for neg_sample in df_current_neg.index.values:
                # Link each pair of positive to a negative
                triplet = (cid,) + each_pair_positive + (neg_sample,)
                triplet_order_arr.append(triplet)
        if len(triplet_order_arr) == 0:
            print(cid)
        return np.array(triplet_order_arr)

    # Retrain FALSE

    # Thread Starting
    pool = ThreadPool()
    threads = [
        pool.apply_async(generate_triplet_sample, (cid,))
        for cid in tqdm(cid_list, desc="[Generate Triplet Dataset] Start thread")
    ]
    # Thread Joining
    result_arr = [
        thread.get()
        for thread in tqdm(threads, desc="[Generate Triplet Dataset] Join thread")
        if thread.get() is not None
    ]
    df_result = pd.DataFrame(np.concatenate(result_arr))
    df_result.rename(columns={0: "cid", 1: "anchor", 2: "pos", 3: "neg"}, inplace=True)
    df_result = shuffle(df_result)
    return df_result


def create_data_loader(loader, batch_size=5000):
    array, lengths = np.array(loader["data"]), np.array(loader["length"])
    data = TensorDataset(
        torch.from_numpy(array).type(torch.LongTensor), torch.ByteTensor(lengths)
    )
    return DataLoader(data, batch_size=batch_size, drop_last=False)


def load_triplet(
        x_padded,
        x_lengths,
        df_triplet_orders,
        batch_size=520,
        retrain=False,
):
    df_triplet_orders = shuffle(df_triplet_orders)
    anc_dict = {"data": [], "length": []}
    pos_dict = {"data": [], "length": []}
    neg_dict = {"data": [], "length": []}
    for row in tqdm(
            df_triplet_orders.iloc[:, :].itertuples(),
            total=df_triplet_orders.shape[0],
            desc="Load triplets",
    ):
        anc_loc, pos_loc, neg_loc = row[2], row[3], row[4]
        anc_dict["data"].append(x_padded[anc_loc])
        anc_dict["length"].append(x_lengths[anc_loc])

        pos_dict["data"].append(x_padded[pos_loc])
        pos_dict["length"].append(x_lengths[pos_loc])

        neg_dict["data"].append(x_padded[neg_loc])
        neg_dict["length"].append(x_lengths[neg_loc])

    anc_loader = create_data_loader(anc_dict, batch_size)
    pos_loader = create_data_loader(pos_dict, batch_size)
    neg_loader = create_data_loader(neg_dict, batch_size)

    return anc_loader, pos_loader, neg_loader
