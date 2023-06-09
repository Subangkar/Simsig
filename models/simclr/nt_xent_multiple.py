import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss_Multiple(nn.Module):
    def __init__(self, temperature=0.5, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, emb: torch.Tensor, adjacency_matrix):
        """
        emb: is batch of embeddings
        z as per SimCLR paper
        """
        batch_size = emb.size(0)
        negatives_mask = (~torch.eye(batch_size, batch_size, dtype=torch.bool, device=emb.device)).float()
        # (b, b) (b, b) -> (b, b)
        positives_mask = adjacency_matrix  # includes comparison with self

        # (b, 128) -> (b, 128)
        z = F.normalize(emb, dim=1)

        # (b, 128) -> (b, 128)
        representations = z

        # (b, 128) -> [(b, 1, 128) * (1, b, 128)] -> (b, b)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        # (b, b) * (b, b) -> (b, b)
        nominator = positives_mask * torch.exp(similarity_matrix / self.temperature)
        # (b, b) * (b, b) -> (b, b)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        # (b) / [(b, b) -> (b)] -> (b)
        loss_partial = -torch.log(torch.sum(nominator, dim=1) / torch.sum(denominator, dim=1))
        # (1)
        loss = torch.mean(loss_partial)
        return loss


if __name__ == "__main__":
    # b = 3
    # l = 3
    # cl = NTXentLoss_Multiple()
    #
    # t1 = torch.randn((b, l))
    # t2 = torch.Tensor([[True, False, True],
    #                    [False, True, False],
    #                    [True, False, True]])
    #
    # cl.forward(t1, t2)
    #
    # t = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])  # , 6], [7, 8, 9]
    # print(torch.diag(t, -2))
    #
    # # (b, 3) -> (2b, 3)
    # representations = torch.cat([torch.Tensor([[1, 2, 3]]), torch.Tensor([[4, 5, 6]])], dim=0)
    # # (2b, 3) -> [(2b, 1, 3) * (1, 2b, 3)] -> (2b, 2b)
    # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    # similarity_matrix = F.cosine_similarity(representations, representations, dim=0)
    # # print(similarity_matrix)
    # t = torch.Tensor([[0.1, 0.4, 0.3], [0.2, 0.1, 0.3]])
    # print(NTXentLoss_Multiple().forward(t, t))
    # b = 4
    # adj_list = [[2, 3],
    #             [],
    #             [0, 3],
    #             [0, 2]]
    # bt = np.array([[101, 102],
    #                [103, 104],
    #                [105, 106],
    #                [107, 108]])
    # ind = torch.Tensor([10, 1, 10, 10])
    # out = []
    # for i, x in enumerate(ind):
    #     out.append(ind == ind[i])
    # out = torch.vstack(out)
    # print(out)
    pass
