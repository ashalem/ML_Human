import torch
import torch.nn as nn

class UniversityMLP(nn.Module):
    """Simple MLP for university decisions"""
    def __init__(self, n_features: int, n_faculties: int):
        super().__init__()
        self.network = nn.Sequential(
            # Add one-hot encoded faculty to features
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_faculties)
        )
    
    def forward(self, x):
        return self.network(x)

class ApplicantMLP(nn.Module):
    """MLP for applicant decisions with softmax output"""
    def __init__(self, n_features: int, n_faculties: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_faculties),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x) 