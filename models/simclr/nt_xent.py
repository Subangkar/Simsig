import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.register_buffer("temperature", torch.tensor(temperature))
        # self.register_buffer("negatives_mask",
        #                      (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=device)).float())

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = emb_i.size(0)
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=emb_i.device)).float()

        # (b, 128) -> (b, 128)
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # (b, 128) -> (2b, 128)
        representations = torch.cat([z_i, z_j], dim=0)
        # (2b, 128) -> [(2b, 1, 128) * (1, 2b, 128)] -> (2b, 2b)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        # (2b, 2b) -> (b)
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        # (b) -> (2b)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # (2b) -> (2b)
        nominator = torch.exp(positives / self.temperature)
        # (2b, 2b) * (2b, 2b) -> (2b, 2b)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        # (2b) / [(2b, 2b) -> (2b)] -> (2b)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        # (1)
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss


if __name__ == "__main__":
    b = 4
    cl = NTXentLoss()

    t1 = torch.randn((b, 128))
    t2 = torch.randn((b, 128))

    cl.forward(t1, t2)

    t = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])  # , 6], [7, 8, 9]
    print(torch.diag(t, -2))

    # (b, 3) -> (2b, 3)
    representations = torch.cat([torch.Tensor([[1, 2, 3]]), torch.Tensor([[4, 5, 6]])], dim=0)
    # (2b, 3) -> [(2b, 1, 3) * (1, 2b, 3)] -> (2b, 2b)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    similarity_matrix = F.cosine_similarity(representations, representations, dim=0)
    # print(similarity_matrix)
    t = torch.Tensor([[0.1, 0.4, 0.3], [0.2, 0.1, 0.3]])
    print(NTXentLoss(2).forward(t, t))
