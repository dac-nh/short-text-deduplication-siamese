def truncate_non_string(X, X_len):
    # Drop rows that have length of word vector = 0
    truncate_index = [i for i in range(0, len(X_len)) if X_len[i] <= 0]
    X, X_len = np.delete(X, truncate_index, axis=0), np.delete(X_len, truncate_index, axis=0)

    return X, X_len, truncate_index