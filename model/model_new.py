import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# --- MODEL --- #
class TripletSiameseModel(torch.nn.Module):
    """
    Triplet Model with embedding
    """

    def __init__(
            self,
            hid_dim=120,
            embeddings=None,
            n_classes=10,
            layers=1,
            bidirectional=True,
    ):
        super(TripletSiameseModel, self).__init__()

        self.embeddings, emb_dim = self._load_embeddings(embeddings)
        self.n_classes = n_classes
        self.gru = torch.nn.GRU(
            emb_dim, hid_dim, layers, batch_first=True, bidirectional=bidirectional
        )
        self.linear_final = torch.nn.Linear(2 * hid_dim, self.n_classes)
        self.linear_distance = torch.nn.Linear(2 * self.n_classes, 1)

    def _load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        word_embeddings = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        #         word_embeddings.weight = torch.nn.Parameter(embeddings)
        emb_dim = embeddings.shape[1]
        return word_embeddings, emb_dim

    def forward(self, x1, x2, x3=None):
        def forward_detail(x):
            x, x_len = x
            x_embed = self.embeddings(x)  # turn off when not use embedding
            x_packed = pack_padded_sequence(
                x_embed, x_len, batch_first=True, enforce_sorted=False
            )
            x_packed, self.hidden_state = self.gru(x_packed)
            output, output_lengths = pad_packed_sequence(x_packed, batch_first=True)
            return self.linear_final(output)

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

            # pos_combination = torch.cat(
            #     [anchors.squeeze(1), positives.squeeze(1)], dim=1
            # )
            # pos = self.linear_distance(pos_combination)
            # pos_loss = F.relu(-self.tanh(pos) + self.margin)
            #
            # neg_combination = torch.cat(
            #     [anchors.squeeze(1), negatives.squeeze(1)], dim=1
            # )
            # neg = self.linear_distance(neg_combination)
            # neg_loss = F.relu(self.tanh(neg) + self.margin)
            # return pos_loss, neg_loss

            return anchors, positives, negatives
        else:
            # Predict purpose
            x1, x2 = forward_detail(x1), forward_detail(x2)
            x1, x2 = torch.sum(x1, dim=1), torch.sum(x2, dim=1)
            return x1, x2

            # # Predict purpose
            # x1, x2 = forward_detail(x1), forward_detail(x2)
            # x1, x2 = torch.sum(x1, dim=1), torch.sum(x2, dim=1)
            # combination = torch.cat([x1.squeeze(1), x2.squeeze(1)], dim=1)
            # res = self.linear_distance(combination)
            # return self.tanh(res)


class TripletDistance(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        self.margin = margin
        super(TripletDistance, self).__init__()

    def forward(self, anchors, positives, negatives, size_average=True):
        anchors_2d = anchors.reshape(anchors.shape[0], -1)
        positives_2d = positives.reshape(positives.shape[0], -1)
        negatives_2d = negatives.reshape(negatives.shape[0], -1)

        similarity_pos = torch.sum(anchors_2d * positives_2d, dim=1) / (
                torch.sqrt(torch.sum(anchors_2d * anchors_2d, dim=1))
                * torch.sqrt(torch.sum(positives_2d * positives_2d, dim=1))
        )

        similarity_neg = torch.sum(anchors_2d * negatives_2d, dim=1) / (
                torch.sqrt(torch.sum(anchors_2d * anchors_2d, dim=1))
                * torch.sqrt(torch.sum(negatives_2d * negatives_2d, dim=1))
        )

        losses = F.relu(-similarity_pos + similarity_neg + self.margin)

        return (
            losses.sum(),
            similarity_pos,
            similarity_neg,
        )
