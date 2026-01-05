# --- START OF FILE sandbox/toposhield.py ---

import torch
import torch.optim as optim
import numpy as np
import networkx as nx
from collections import defaultdict

from sandbox.st_gat_model import GSTNet
from sandbox.feature_extractor import FeatureExtractor

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("[!] Warning: 'python-louvain' not installed. Community defense disabled.")


class TopoSHIELD:
    """
    Implements the TopoSHIELD defense mechanism:
    1. Perception (FeatureExtractor)
    2. Risk Prediction (GSTNet)
    3. Multi-level Defense (Graph Pruning, Anchor, Community Isolation)
    4. Online Learning
    """
    def __init__(
        self,
        num_agents,
        device='cpu',
        hidden_dim=64,
        history_window=5,
        use_anchor: bool = True,
        use_community_isolation: bool = True
    ):
        self.num_agents = num_agents
        self.device = device
        self.use_anchor = use_anchor
        self.use_community_isolation = use_community_isolation

        self.extractor = FeatureExtractor(num_agents, device=device, history_window=history_window)

        self.node_in_dim = 6
        self.edge_in_dim = 3

        self.model = GSTNet(
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=2
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.loss_fn = torch.nn.MSELoss()

        self.current_graph_data = None

        self.min_affinity_to_keep = 0.35
        self.isolation_threshold = 0.85
        self.community_risk_threshold = 0.75

        self.anchor_id = 0
        self.initial_topo = None

    def register_initial_topology(self, topo_list):
        self.initial_topo = topo_list

    def step(self, agent_list, current_step_log):
        """
        Main execution loop: Perception -> Prediction -> Defense Decision.
        Returns the updated topology map.
        """
        try:
            X, Z = self._build_gst_from_log(agent_list, current_step_log)
            self.current_graph_data = (X, Z)

            self.model.eval()
            with torch.no_grad():
                nre_scores, eac_affinity = self.model(X, Z)

            new_topology = self._apply_multilevel_defense(
                nre_scores=nre_scores,
                eac_affinity=eac_affinity
            )
            return new_topology

        except Exception as e:
            print(f"[TopoSHIELD Error] {e}. Using fallback.")
            return self._get_fallback_topology()

    def update_model(self, step_score):
        """
        Online learning update using step_score as a weak supervision signal.
        """
        if self.current_graph_data is None:
            return

        try:
            X, Z = self.current_graph_data
            risk_signal = max(0.0, min(1.0, 1.0 - (float(step_score) / 10.0)))
            target = torch.full((self.num_agents, 1), risk_signal, device=self.device)

            self.model.train()
            self.optimizer.zero_grad()
            pred_nre, _ = self.model(X, Z)

            loss = self.loss_fn(pred_nre, target)
            loss.backward()
            self.optimizer.step()

        except Exception as e:
            print(f"[Update Error] {e}")

    def _build_gst_from_log(self, agent_list, logs):
        """
        Extracts recent content from logs, updates feature extractor, and builds GST tensors.
        """
        last_msg = {i: "" for i in range(self.num_agents)}

        for entry in reversed(logs or []):
            try:
                content = entry.get("content", "")
            except Exception:
                continue

            if not content:
                continue

            who = None
            if isinstance(entry, dict):
                if "name" in entry:
                    who = entry.get("name")
                elif "send" in entry:
                    who = entry.get("send")

            if who is None:
                continue

            try:
                who_int = int(who)
            except Exception:
                continue

            if 0 <= who_int < self.num_agents and not last_msg.get(who_int):
                last_msg[who_int] = content

        for i in range(self.num_agents):
            if not last_msg[i]:
                try:
                    last_msg[i] = self.extractor.agent_history[i]["last_content"]
                except Exception:
                    last_msg[i] = ""

        for agent_idx, agent in enumerate(agent_list):
            my_content = last_msg.get(agent_idx, "")

            neighbors_content = []
            nb_list = []
            try:
                nb_list = list(getattr(agent.background, "neighbors", []) or [])
            except Exception:
                nb_list = []

            for nb in nb_list:
                try:
                    nb_int = int(nb)
                except Exception:
                    continue
                if 0 <= nb_int < self.num_agents and nb_int != agent_idx:
                    neighbors_content.append(last_msg.get(nb_int, ""))

            self.extractor.update_agent_step(agent_idx, my_content, neighbors_content)

        for src in range(self.num_agents):
            src_content = last_msg.get(src, "")
            for dst in range(self.num_agents):
                if src == dst:
                    continue
                dst_content = last_msg.get(dst, "")
                self.extractor.update_edge_step(src, dst, src_content, dst_content)

        return self.extractor.build_gst_tensors()

    def _apply_multilevel_defense(self, nre_scores, eac_affinity):
        """
        Applies defense logic: Anchor selection, Community Isolation, and Micro Pruning.
        """
        nre_vals = nre_scores.squeeze(-1).detach().cpu().tolist()
        if not isinstance(nre_vals, list):
            nre_vals = [float(nre_vals)]

        eac_mat = eac_affinity.squeeze(-1).detach().cpu().numpy()

        if self.use_anchor:
            self.anchor_id = int(np.argmin(nre_vals)) if len(nre_vals) else 0
        else:
            self.anchor_id = None

        risky_nodes = [
            i for i, r in enumerate(nre_vals)
            if float(r) > self.isolation_threshold
        ]

        new_topo = {i: [] for i in range(self.num_agents)}

        communities = {}
        blocked_communities = []

        if self.use_community_isolation and HAS_LOUVAIN:
            G = nx.Graph()
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    w = max(float(eac_mat[i, j]), 1e-4)
                    G.add_edge(i, j, weight=w)

            if G.number_of_nodes() > 0:
                partition = community_louvain.best_partition(G, weight="weight")
                communities = partition

                comm_risks = defaultdict(list)
                for node, comm in partition.items():
                    if node < len(nre_vals):
                        comm_risks[comm].append(float(nre_vals[node]))

                for comm, risks in comm_risks.items():
                    if np.mean(risks) > self.community_risk_threshold:
                        blocked_communities.append(comm)

        for src in range(self.num_agents):
            for dst in range(self.num_agents):
                if src == dst:
                    continue

                if self.use_community_isolation and communities:
                    sc = communities.get(src)
                    dc = communities.get(dst)
                    if sc != dc and (sc in blocked_communities or dc in blocked_communities):
                        continue

                if self.use_anchor and src in risky_nodes:
                    if dst == self.anchor_id:
                        new_topo[src].append(dst)
                    continue

                affinity = float(eac_mat[src, dst])
                if affinity < self.min_affinity_to_keep:
                    self.extractor.record_attack(src, dst)
                    continue

                new_topo[src].append(dst)

        if self.use_anchor:
            for i in range(self.num_agents):
                if i in risky_nodes:
                    if not new_topo[i] and i != self.anchor_id:
                        new_topo[i] = [self.anchor_id]
                    continue

                if not new_topo[i] and i != self.anchor_id:
                    new_topo[i].append(self.anchor_id)

        return new_topo

    def _get_fallback_topology(self):
        return {i: [j for j in range(self.num_agents) if j != i] for i in range(self.num_agents)}

# --- END OF FILE sandbox/toposhield.py ---