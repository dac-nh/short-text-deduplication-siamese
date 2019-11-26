import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """

    def __init__(
            self,
            gru_hid_dim=120,
            d_a=100,
            r=10,
            embeddings=None,
            n_classes=50,
            margin=0.2,
            cuda=None,
    ):
        """
        Initializes parameters suggested in paper

        Args:
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes

        Returns:
            self

        Raises:
            Exception
        """
        super(StructuredSelfAttention, self).__init__()

        self.embeddings, emb_dim = self._load_embeddings(embeddings)
        self.embeddings.weight.require_gradient = False
        self.gru = torch.nn.GRU(emb_dim, gru_hid_dim, 1, batch_first=True)
        self.linear_first = torch.nn.Linear(gru_hid_dim, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(gru_hid_dim, self.n_classes)
        self.gru_hid_dim = gru_hid_dim
        self.r = r
        self.tanh = torch.nn.Tanh()
        self.device = cuda
        self.margin = margin

    def _load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        word_embeddings = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        #         word_embeddings.weight = torch.nn.Parameter(embeddings)
        emb_dim = embeddings.shape[1]

        return word_embeddings, emb_dim

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n

        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors

        """

        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=0)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, x1, x2, x3=None):
        def forward_detail(x):
            x, x_len = x
            embeddings = self.embeddings(x)
            x_packed = pack_padded_sequence(
                embeddings, x_len, batch_first=True, enforce_sorted=False
            )
            x_packed, self.hidden_state = self.gru(x_packed)
            x_padded, output_lengths = pad_packed_sequence(x_packed, batch_first=True)
            x = torch.tanh(self.linear_first(x_padded))
            x = self.linear_second(x)
            x = self.softmax(x, 1)
            attention = x.transpose(1, 2)
            sentence_embeddings = attention @ x_padded
            avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r
            return F.log_softmax(self.linear_final(avg_sentence_embeddings), dim=0), attention

        if x3 is not None:
            # Training purpose - Triplet input

            (anchors, _), (positives, _), (negatives, _) = (
                forward_detail(x1),
                forward_detail(x2),
                forward_detail(x3),
            )

            return anchors, positives, negatives
        else:
            # Predict purpose
            (x1, att1), (x2, att2) = forward_detail(x1), forward_detail(x2)
            return x1, x2
