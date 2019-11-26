import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class TripletPredict(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, embeddings, margin):
        super(TripletLoss, self).__init__()

    def forward(self, anchors, positives, negatives, size_average=True):
        anchors_2d = anchors.reshape(anchors.shape[0], -1)
        positives_2d = positives.reshape(positives.shape[0], -1)
        negatives_2d = negatives.reshape(negatives.shape[0], -1)

        # Euclidean distance
        # distance_positive = (anchors_2d - positives_2d).pow(2).sum(1).pow(.5)
        # distance_negative = (anchors_2d - negatives_2d).pow(2).sum(1).pow(.5)
        # losses = f.relu(distance_positive - distance_negative + self.margin)

        # https://cmry.github.io/notes/euclidean-v-cosine
        # cosine distance: batch_size x embedded_data_size
        distance_positive = (anchors_2d @ positives_2d.T)[0] / (
            (anchors_2d @ anchors_2d.T)[0] * (positives_2d @ positives_2d.T)[0]
        )
        distance_negative = (anchors_2d @ negatives_2d.T)[0] / (
            (anchors_2d @ anchors_2d.T)[0] * (negatives_2d @ negatives_2d.T)[0]
        )

        losses = f.relu(distance_positive - distance_negative + self.margin)

        return (
            losses.mean() if size_average else losses.sum(),
            distance_positive,
            distance_negative,
        )


# --- MODEL --- #
class TripletModel(torch.nn.Module):
    """
    Triplet Model with embedding
    """
    def __init__(
        self,
        lstm_hid_dim=120,
        max_len=40,
        embeddings=None,
        n_classes=50,
        margin=0.2,
        cuda=None,
    ):
        super(TripletModel, self).__init__()

        self.embeddings, emb_dim = self._load_embeddings(embeddings)
        self.lstm = torch.nn.LSTM(emb_dim, lstm_hid_dim, 1, batch_first=True, bidirectional=True)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(2*lstm_hid_dim, self.n_classes)
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.linear_distance = torch.nn.Linear(self.n_classes * 2, 1)
        self.tanh = torch.nn.Tanh()
        self.device = cuda
        self.margin = margin

    def _load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
#         word_embeddings.weight = torch.nn.Parameter(embeddings)
        emb_dim = embeddings.size(1)

        return word_embeddings, emb_dim

    def init_hidden(self):
        if self.device is None:
#             return (
#                 Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)),
#                 Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)),
#             )
            # Bidirectional
            return (
                Variable(torch.zeros(2, self.batch_size, self.lstm_hid_dim)),
                Variable(torch.zeros(2, self.batch_size, self.lstm_hid_dim)),
            )
#         return (
#             Variable(
#                 torch.zeros(1, self.batch_size, self.lstm_hid_dim).cuda(self.device)
#             ),
#             Variable(
#                 torch.zeros(1, self.batch_size, self.lstm_hid_dim).cuda(self.device)
#             ),
#         )
        # Bidirectional
        return (
            Variable(
                torch.zeros(2, self.batch_size, self.lstm_hid_dim).cuda(self.device)
            ),
            Variable(
                torch.zeros(2, self.batch_size, self.lstm_hid_dim).cuda(self.device)
            ),
        )

    def forward(self, x1, x2, x3=None):
        self.batch_size = x1.shape[0]
        self.hidden_state = self.init_hidden()

        def forward_detail(x):
            embeddings = self.embeddings(x)  # turn off when not use embedding
            outputs, self.hidden_state = self.lstm(
                embeddings.view(self.batch_size, self.max_len, -1))
            return self.linear_final(outputs)

        if x3 is not None:
            # Training purpose - Triplet input
            anchors, positives, negatives = (
                forward_detail(x1),
                forward_detail(x2),
                forward_detail(x3),
            )
            anchors, positives, negatives = (
                torch.sum(anchors, dim=1),
                torch.sum(positives, dim=1),
                torch.sum(negatives, dim=1),
            )
            pos_combination = torch.cat(
                [anchors.squeeze(1), positives.squeeze(1)], dim=1
            )
            pos = self.linear_distance(pos_combination)
            pos_loss = F.relu(-self.tanh(pos) + self.margin)

            neg_combination = torch.cat(
                [anchors.squeeze(1), negatives.squeeze(1)], dim=1
            )
            neg = self.linear_distance(neg_combination)
            neg_loss = F.relu(self.tanh(neg) + self.margin)
            return pos_loss, neg_loss
        else:
            # Predict purpose
            x1, x2 = forward_detail(x1), forward_detail(x2)
            x1, x2 = torch.sum(x1, dim=1), torch.sum(x2, dim=1)
            combination = torch.cat([x1.squeeze(1), x2.squeeze(1)], dim=1)
            res = self.linear_distance(combination)
            return self.tanh(res)
        
        
class TripletNoEmbeddingModel(torch.nn.Module):
    """
    Triplet model without embedding inside
    """
    def __init__(
        self,
        lstm_hid_dim=120,
        max_len=40,
        n_classes=50,
        margin=0.2,
        cuda=None,
        emb_dim=300
    ):
        super(TripletNoEmbeddingModel, self).__init__()

        self.lstm = torch.nn.LSTM(emb_dim, lstm_hid_dim, 1, batch_first=True)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(lstm_hid_dim, self.n_classes)
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.linear_distance = torch.nn.Linear(self.n_classes * 2, 1)
        self.tanh = torch.nn.Tanh()
        self.device = cuda
        self.margin = margin

    def init_hidden(self):
        if self.device is None:
            return (
                Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)),
                Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)),
            )

        return (
            Variable(
                torch.zeros(1, self.batch_size, self.lstm_hid_dim).cuda(self.device)
            ),
            Variable(
                torch.zeros(1, self.batch_size, self.lstm_hid_dim).cuda(self.device)
            ),
        )

    def forward(self, x1, x2, x3=None):
        self.batch_size = x1.shape[0]
        self.hidden_state = self.init_hidden()

        def forward_detail(x):
            outputs, self.hidden_state = self.lstm(
                x.view(self.batch_size, self.max_len, -1), self.hidden_state
            )
            return self.linear_final(outputs)

        if x3 is not None:
            # Training purpose - Triplet input
            anchors, positives, negatives = (
                forward_detail(x1),
                forward_detail(x2),
                forward_detail(x3),
            )
            anchors, positives, negatives = (
                torch.sum(anchors, dim=1),
                torch.sum(positives, dim=1),
                torch.sum(negatives, dim=1),
            )
            pos_combination = torch.cat(
                [anchors.squeeze(1), positives.squeeze(1)], dim=1
            )
            pos = self.linear_distance(pos_combination)
            pos_loss = F.relu(-self.tanh(pos) + self.margin)

            neg_combination = torch.cat(
                [anchors.squeeze(1), negatives.squeeze(1)], dim=1
            )
            neg = self.linear_distance(neg_combination)
            neg_loss = F.relu(self.tanh(neg) + self.margin)
            return pos_loss, neg_loss
        else:
            # Predict purpose
            x1, x2 = forward_detail(x1), forward_detail(x2)
            x1, x2 = torch.sum(x1, dim=1), torch.sum(x2, dim=1)
            combination = torch.cat([x1.squeeze(1), x2.squeeze(1)], dim=1)
            res = self.linear_distance(combination)
            return self.tanh(res)
        
        
class TripletBoWModel(torch.nn.Module):
    """
    Triplet bag of word model
    """
    def __init__(
        self,
        lstm_hid_dim=120,
        max_len=40,
        n_classes=50,
        margin=0.2,
        cuda=None
    ):
        super(TripletBoWModel, self).__init__()
        
#         self.conv = torch.nn.Conv(self.max_length, 1, kernel_size=)
        self.lstm = torch.nn.LSTM(1, lstm_hid_dim, batch_first=True)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(lstm_hid_dim, self.n_classes)
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.linear_distance = torch.nn.Linear(self.n_classes * 2, 1)
        self.tanh = torch.nn.Tanh()
        self.device = cuda
        self.margin = margin

    def init_hidden(self):
        if self.device is None:
            return (
                Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)),
                Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)),
            )

        return (
            Variable(
                torch.zeros(1, self.batch_size, self.lstm_hid_dim).cuda(self.device)
            ),
            Variable(
                torch.zeros(1, self.batch_size, self.lstm_hid_dim).cuda(self.device)
            ),
        )

    def forward(self, x1, x2, x3=None):
        self.batch_size = x1.shape[0]
        self.hidden_state = self.init_hidden()

        def forward_detail(x):
            outputs, self.hidden_state = self.lstm(
                x.view(self.batch_size, self.max_len, -1), self.hidden_state)
            return self.linear_final(outputs)

        if x3 is not None:
            # Training purpose - Triplet input
            anchors, positives, negatives = (
                forward_detail(x1),
                forward_detail(x2),
                forward_detail(x3),
            )
            anchors, positives, negatives = (
                torch.sum(anchors, dim=1),
                torch.sum(positives, dim=1),
                torch.sum(negatives, dim=1),
            )
            pos_combination = torch.cat(
                [anchors.squeeze(1), positives.squeeze(1)], dim=1
            )
            pos = self.linear_distance(pos_combination)
            pos_loss = F.relu(-self.tanh(pos) + self.margin)

            neg_combination = torch.cat(
                [anchors.squeeze(1), negatives.squeeze(1)], dim=1
            )
            neg = self.linear_distance(neg_combination)
            neg_loss = F.relu(self.tanh(neg) + self.margin)
            return pos_loss, neg_loss
        else:
            # Predict purpose
            x1, x2 = forward_detail(x1), forward_detail(x2)
            x1, x2 = torch.sum(x1, dim=1), torch.sum(x2, dim=1)
            combination = torch.cat([x1.squeeze(1), x2.squeeze(1)], dim=1)
            res = self.linear_distance(combination)
            return self.tanh(res)