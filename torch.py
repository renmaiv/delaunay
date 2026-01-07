import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(HypergraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, H):
        # H: incidence matrix (num_nodes x num_hyperedges)
        Dv = torch.diag(torch.pow(H.sum(1), -0.5))  # node degrees
        De = torch.diag(torch.pow(H.sum(0), -1.0))  # edge degrees
        H_T = H.t()
        HTHT = torch.matmul(torch.matmul(Dv, H), torch.matmul(De, H_T))
        return self.linear(torch.matmul(HTHT, X))

class HGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(HGNN, self).__init__()
        self.hgc1 = HypergraphConvolution(in_dim, hidden_dim)
        self.hgc2 = HypergraphConvolution(hidden_dim, out_dim)

    def forward(self, X, H):
        X = F.relu(self.hgc1(X, H))
        X = self.hgc2(X, H)
        return X

# Example usage:
X = torch.randn(100, 16)        # 100 nodes with 16 features
H = torch.randint(0, 2, (100, 30)).float()  # Incidence matrix: 100 nodes, 30 hyperedges
model = HGNN(16, 32, 10)        # Output: 10 classes
output = model(X, H)
print(output.shape)  # (100, 10)
