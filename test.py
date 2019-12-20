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


def data_loader(test_df_1, test_df_2, embedding_index):
    # Data Preparation pipeline
    # Create Dataloader based on two dataframe have row 'content' in it
    X1, X1_lens = load_padded_data(pd.DataFrame(test_df_1), embedding_index)
    X2, X2_lens = load_padded_data(pd.DataFrame(test_df_2), embedding_index)

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


def validate(model, X1, X2, device):
    y_true = []
    y_pred = []
    dist_list = []
    X1_embed, X2_embed = [], []
    for a, b in zip(X1, X2):
        # Send data to graphic card - Cuda0
        a, b = to_cuda(a, device), to_cuda(b, device)
        with torch.no_grad():
            a, b = model(a, b)
            a, b = a.cpu(), b.cpu()
            a = a.reshape(a.shape[0], -1)
            b = b.reshape(b.shape[0], -1)
            #         att1 = att1.cpu()
            #         att2 = att2.cpu()
            X1_embed.append(a)
            X2_embed.append(b)
            dist = np.array(
                [
                    cosine_similarity([a[i].numpy()], [b[i].numpy()])
                    for i in range(0, len(a))
                ]
            ).flatten()
            dist_list.append(dist)

            y_true_curr = np.zeros(len(dist))
            y_true = np.concatenate([y_true, y_true_curr])

            y_pred_curr = np.ones(len(dist))
            y_pred_curr[np.where(dist < 0.74)[0]] = 0
            y_pred = np.concatenate([y_pred, y_pred_curr])
    y_true[1176:] = 1
    return y_true, y_pred, dist_list, X1_embed, X2_embed


# ---- MAIN

path = "/data/dac/dedupe-project/test/"
test_df = pd.read_csv(path + "test_address_3.csv", encoding="ISO-8859-1")
test_df.fillna("", inplace=True)
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

test_df_1a, test_df_1b = create_test(1176, test_df_1, test_df_2)
test_X1, test_X2, test_drop = data_loader(test_df_1a, test_df_1b)

# ---- Load model, distance and optimizer
model, distance, optimizer = load_triplet_siamese_model("/data/dac/dedupe-project/new/model/triplet_siamese_50d_bi_gru_random", embedding_index, 50)
model.eval()

# Test
y_true, y_pred, _, _, _ = validate(model, test_X1, test_X2, device)
print(
    "\rBatch:\t{}\tLoss:\t{}\tAccuracy:\t{}\tF1-score:\t{}\t".format(
        batch,
        round(float(loss), 4),
        round(accuracy_score(y_true, y_pred), 4),
        round(f1_score(y_true, y_pred), 4),
    ),
    end="",
)
print(
    "Precision:\t{}\t\tRecall:\t{}".format(
        round(precision_score(y_true, y_pred), 4),
        round(recall_score(y_true, y_pred), 4),
    ),
    end="",
)