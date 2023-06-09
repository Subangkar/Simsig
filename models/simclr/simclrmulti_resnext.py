import torch.nn as nn

from models.resnext1d import ResNext1D


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SimCLRMulti_Resnext(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, projection_dim: int, **kwargs):
        super().__init__()

        self.encoder = ResNext1D()
        self.n_features = self.encoder.fc.in_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

        print("Representation Dim: ", self.n_features, "Projection Dim: ", projection_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
