# --- START OF FILE sandbox/st_gat_model.py ---

import torch
import torch.nn as nn


class GSTLayer(nn.Module):
    """
    Graph Spatial-Temporal (GST) Layer.
    
    Mechanisms:
      1. Edge Gating (eta): Computes attention/gate values based on edge features.
      2. Aggregation: Aggregates messages from source nodes weighted by the edge gate.
      3. Update: Updates node hidden states based on aggregated messages and previous states.
      4. Affinity: Computes edge affinity scores based on node pairs and edge features.

    Input:
      H_t: (N, Dh) - Node hidden states at time t.
      Z_t: (N, N, De) - Edge features at time t.

    Output:
      H_next: (N, Dh) - Updated node hidden states.
      E_aff: (N, N) - Calculated edge affinity matrix.
    """
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_affinity_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, H_t: torch.Tensor, Z_t: torch.Tensor):
        """
        Forward pass for the GST Layer.
        """
        N = H_t.size(0)
        eta = self.edge_gate(Z_t)
        src_states = H_t.unsqueeze(1).expand(N, N, self.hidden_dim)
        msg = self.msg_mlp(src_states)
        gated = eta * msg  # (N, N, Dh)
        agg = gated.sum(dim=0)
        H_next = self.update_mlp(torch.cat([H_t, agg], dim=-1))  # (N, Dh)

        dst_states = H_t.unsqueeze(0).expand(N, N, self.hidden_dim)

        edge_in = torch.cat([src_states, dst_states, Z_t], dim=-1)  # (N, N, 2*Dh + De)
        E_aff = self.edge_affinity_head(edge_in).squeeze(-1)  # (N, N)

        return H_next, E_aff


class GSTNet(nn.Module):
    """
    Graph Spatial-Temporal Network (GSTNet).
    
    Processes temporal sequences of node and edge features to detect risks and calculate topology affinity.

    Input Tensors:
      X: (N, T, Dnode) - Node features over time.
      Z: (N, N, T, Dedge) - Edge features over time.

    Output Tensors:
      nre_scores: (N, 1) - Node Risk Estimation scores.
      eac_affinity: (N, N, 1) - Edge Affinity Consensus scores.
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        temporal_kernel: int = 3,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gst_layers = nn.ModuleList([
            GSTLayer(hidden_dim=hidden_dim, edge_dim=edge_in_dim)
            for _ in range(num_layers)
        ])

        self.decay_logit = nn.Parameter(torch.tensor(1.0))

        padding = (temporal_kernel - 1) // 2
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=temporal_kernel,
            padding=padding
        )

        self.nre_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.final_edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X: torch.Tensor, Z: torch.Tensor):
        """
        Forward pass for GSTNet.
        
        Args:
            X: Node features (N, T, Dnode)
            Z: Edge features (N, N, T, Dedge)
        """
        assert X.dim() == 3, f"X must be (N, T, Dnode), got {tuple(X.shape)}"
        assert Z.dim() == 4, f"Z must be (N, N, T, Dedge), got {tuple(Z.shape)}"

        N, T, _ = X.shape
        decay = torch.sigmoid(self.decay_logit)  

        H0_seq = self.node_proj(X)

        H_list = []
        E_last = None

        H_prev = torch.zeros((N, self.hidden_dim), device=X.device, dtype=X.dtype)

        for t in range(T):
            Z_t = Z[:, :, t, :]       # (N, N, De)
            H_t = H0_seq[:, t, :]     # (N, Dh)
            H_tilde = decay * H_prev + (1.0 - decay) * H_t

            H_now = H_tilde
            E_aff = None
            for layer in self.gst_layers:
                H_now, E_aff = layer(H_now, Z_t)

            H_list.append(H_now)
            H_prev = H_now
            E_last = E_aff

        H_seq = torch.stack(H_list, dim=1)

        H_conv = self.temporal_conv(H_seq.transpose(1, 2)).transpose(1, 2)

        H_last = H_conv[:, -1, :]  # (N, Dh)
        nre_scores = self.nre_head(H_last)  # (N, 1)

        if E_last is None:
            E_last = torch.zeros((N, N), device=X.device, dtype=X.dtype)

        eac_affinity = E_last.unsqueeze(-1)  # (N, N, 1)

        return nre_scores, eac_affinity

# --- END OF FILE sandbox/st_gat_model.py ---