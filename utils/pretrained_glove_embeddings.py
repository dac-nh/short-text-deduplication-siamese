import numpy as np
import torch
from tqdm import tqdm_notebook as tqdm
import os
import pickle
 
main_path =  "/data/dac/dedupe-project/"
def load_glove_embeddings(
    word2idx,
    embedding_dim=50,
    path="embedding/glove.840B.300d.txt",
    dump_path="embedding/embedded_data.pkl",
    retrain=False,
):
    """
    Loading the glove embeddings for exist words only
    """
    dump_path = main_path + dump_path
    if os.path.isfile(dump_path) and not retrain:
        # Load pretrained model
        embeddings = pickle.load(open(dump_path, "rb"))
        return embeddings
    # Load file and embedding data
    path = main_path + path
    embeddings = np.zeros([len(word2idx), embedding_dim])
    with open(path, encoding="utf8") as f:
        for line in tqdm(f.readlines(), desc="Load embedding"):
            # Append embedding of exist word vocabulary
            values = line.split()
            word = "".join(values[:-300])
            vector = np.array(values[-300:], dtype="float32")
            index = word2idx.get(word)

            if index is not None:
                embeddings[index] = vector
        embeddings = torch.from_numpy(np.array(embeddings)).float()
        pickle.dump(embeddings, open(dump_path, "wb"))
        print("Load pretrained embedding successfully!")
        return embeddings