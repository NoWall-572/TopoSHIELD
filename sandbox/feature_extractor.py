# --- START OF FILE sandbox/feature_extractor.py ---

import torch
import numpy as np
import re
from collections import deque, defaultdict
from sandbox.utils import get_embedding, cosine_similarity


class FeatureExtractor:
    """
    Extracts features for the Graph Spatial-Temporal (GST) model.
    
    Output Tensors:
      X: (N, T, 6) - Node Features
      Z: (N, N, T, 3) - Edge Features

    Node Features (6 dimensions):
      F1: Rejection Rate (Consensus divergence from neighbors)
      F2: Sensitive Triggers (Presence of suspicious keywords)
      F3: Anomaly Z-Score (Statistical deviation in message frequency)
      F4: Embedding Volatility (Semantic shift from previous state)
      F5: Contradiction Rate (NLI-style consistency check)
      F6: Resource Usage (Token consumption proxy)

    Edge Features (3 dimensions):
      E1: Semantic Similarity (Contextual alignment)
      E2: Attack History (Historical malicious interactions)
      E3: Bandwidth (Message volume intensity)
    """

    def __init__(self, num_agents, device='cpu', history_window=5):
        self.num_agents = num_agents
        self.device = device
        self.window = history_window

        self.agent_history = defaultdict(lambda: {
            "node_feats": deque(maxlen=history_window),
            "contents": deque(maxlen=history_window),
            "embeddings": deque(maxlen=history_window),
            "msg_counts": deque(maxlen=history_window),
            "rejection_flags": deque(maxlen=history_window),
            "contradiction_flags": deque(maxlen=history_window),
            "last_content": ""
        })

        self.edge_history = defaultdict(lambda: {
            "edge_feats": deque(maxlen=history_window),
            "attack_count": 0,
            "msg_volume": deque(maxlen=history_window)
        })

        self.sensitive_patterns = [
            r"ignore previous", r"jailbreak", r"system prompt",
            r"attack", r"misinfo", r"don't trust", r"hack", r"override",
            r"cannot", r"unable to"
        ]

        # NLI-style negation patterns for contradiction detection
        self.negation_patterns = [
            r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bnone\b",
            r"\bcannot\b", r"\bcan't\b", r"\bdon't\b", r"\bdoesn't\b",
            r"\bisn't\b", r"\baren't\b", r"\bwon't\b", r"\bwithout\b",
            r"不是", r"不能", r"无法", r"不要", r"别"
        ]

    @staticmethod
    def _tokenize_simple(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)
        toks = [t for t in text.split() if len(t) >= 2]
        return toks

    def _has_negation(self, text: str) -> int:
        if not text:
            return 0
        for p in self.negation_patterns:
            if re.search(p, text, re.IGNORECASE):
                return 1
        return 0

    def _contradiction_score(self, prev_text: str, curr_text: str, prev_emb: np.ndarray, curr_emb: np.ndarray) -> float:
        """
        Calculates a heuristic contradiction score in range [0, 1].
        
        Logic:
          - High token overlap indicates the same topic.
          - High semantic distance (1 - cosine_sim) indicates divergence.
          - 'Negation flip' indicates a potential logical reversal.
        """
        if not prev_text or not curr_text:
            return 0.0

        pt = set(self._tokenize_simple(prev_text))
        ct = set(self._tokenize_simple(curr_text))
        if not pt or not ct:
            overlap = 0.0
        else:
            overlap = len(pt & ct) / (min(len(pt), len(ct)) + 1e-6)

        sim = cosine_similarity(curr_emb, prev_emb) if (isinstance(curr_emb, np.ndarray) and isinstance(prev_emb, np.ndarray)) else 0.0
        sem_div = max(0.0, min(1.0, (1.0 - float(sim))))

        neg_flip = 1.0 if (self._has_negation(prev_text) ^ self._has_negation(curr_text)) else 0.0
      
        score = (0.45 * sem_div) + (0.35 * neg_flip * overlap) + (0.20 * overlap * sem_div)
        return float(max(0.0, min(1.0, score)))

    def update_agent_step(self, agent_id: int, content: str, neighbors_content=None):
        if neighbors_content is None:
            neighbors_content = []

        hist = self.agent_history[agent_id]
        content = content or ""

        if content.strip():
            curr_emb = get_embedding(content)
            if curr_emb is None or not isinstance(curr_emb, np.ndarray):
                curr_emb = np.zeros(1536)
        else:
            curr_emb = np.zeros(1536)

        rej_flag = 0
        if neighbors_content and content.strip():
            nb_embs = []
            for n in neighbors_content:
                if n and str(n).strip():
                    e = get_embedding(n)
                    if isinstance(e, np.ndarray):
                        nb_embs.append(e)
            if nb_embs:
                avg_nb = np.mean(np.array(nb_embs), axis=0)
                sim_nb = cosine_similarity(curr_emb, avg_nb)
                if sim_nb < 0.3:
                    rej_flag = 1

        msg_count = 1 if content.strip() else 0

        trig = 0
        if content.strip():
            trig = sum(1 for p in self.sensitive_patterns if re.search(p, content, re.IGNORECASE))
        f2_sensitive = min(trig / 3.0, 1.0)

        tmp_counts = list(hist["msg_counts"]) + [msg_count]
        if len(tmp_counts) > 2:
            mean = float(np.mean(tmp_counts))
            std = float(np.std(tmp_counts)) + 1e-6
            z = abs((msg_count - mean) / std)
            f3_anomaly = min(z / 3.0, 1.0)
        else:
            f3_anomaly = 0.0

        f4_vol = 0.0
        prev_text = ""
        prev_emb = None
        if len(hist["embeddings"]) >= 1:
            prev_emb = np.array(hist["embeddings"][-1])
            prev_text = hist["last_content"]
            sim_prev = cosine_similarity(curr_emb, prev_emb)
            f4_vol = float(1.0 - sim_prev)

        contra_flag = 0
        if prev_emb is not None and prev_text:
            cscore = self._contradiction_score(prev_text, content, prev_emb, curr_emb)
            contra_flag = 1 if cscore >= 0.60 else 0

        tmp_rej = list(hist["rejection_flags"]) + [rej_flag]
        f1_rej = float(np.mean(tmp_rej)) if tmp_rej else 0.0

        tmp_contra = list(hist["contradiction_flags"]) + [contra_flag]
        f5_contra = float(np.mean(tmp_contra)) if tmp_contra else 0.0

        token_len = len(content.split()) if content.strip() else 0
        f6_res = min(token_len / 500.0, 1.0)

        node_feat = torch.tensor(
            [f1_rej, f2_sensitive, f3_anomaly, f4_vol, f5_contra, f6_res],
            dtype=torch.float,
            device=self.device
        )

        hist["node_feats"].append(node_feat)
        hist["contents"].append(content)
        hist["embeddings"].append(curr_emb)
        hist["msg_counts"].append(msg_count)
        hist["rejection_flags"].append(rej_flag)
        hist["contradiction_flags"].append(contra_flag)
        hist["last_content"] = content


    def update_edge_step(self, src_id: int, dst_id: int, msg_content: str, dst_context: str):
        key = f"{src_id}->{dst_id}"
        hist = self.edge_history[key]

        msg_content = msg_content or ""
        dst_context = dst_context or ""


        e1_sim = 0.0
        if msg_content.strip() and dst_context.strip():
            emb_msg = get_embedding(msg_content)
            emb_ctx = get_embedding(dst_context)
            if isinstance(emb_msg, np.ndarray) and isinstance(emb_ctx, np.ndarray):
                sim = cosine_similarity(emb_msg, emb_ctx)
                e1_sim = float((sim + 1.0) / 2.0)

        e2_hist = min(hist["attack_count"] / 5.0, 1.0)

        cur_len = len(msg_content.split()) if msg_content.strip() else 0
        hist["msg_volume"].append(cur_len)
        avg_vol = float(np.mean(hist["msg_volume"])) if hist["msg_volume"] else 0.0
        e3_bw = min(avg_vol / 100.0, 1.0)

        edge_feat = torch.tensor([e1_sim, e2_hist, e3_bw], dtype=torch.float, device=self.device)
        hist["edge_feats"].append(edge_feat)

    def record_attack(self, src, dst):
        key = f"{src}->{dst}"
        self.edge_history[key]["attack_count"] += 1

    def build_gst_tensors(self):
        """
        Constructs the batched tensors for the model.
        Returns:
          X: (N, T, 6) - Node features with time dimension.
          Z: (N, N, T, 3) - Edge features with time dimension.
        Padding:
          Left-padded with zeros if history is shorter than the window T.
        """
        N = self.num_agents
        T = self.window
        Dn = 6
        De = 3

        X = torch.zeros((N, T, Dn), dtype=torch.float, device=self.device)
        Z = torch.zeros((N, N, T, De), dtype=torch.float, device=self.device)

        for i in range(N):
            feats = list(self.agent_history[i]["node_feats"])
            start = T - len(feats)
            if start < 0:
                feats = feats[-T:]
                start = 0
            for k, f in enumerate(feats):
                X[i, start + k, :] = f

        for src in range(N):
            for dst in range(N):
                if src == dst:
                    continue
                key = f"{src}->{dst}"
                feats = list(self.edge_history[key]["edge_feats"])
                start = T - len(feats)
                if start < 0:
                    feats = feats[-T:]
                    start = 0
                for k, f in enumerate(feats):
                    Z[src, dst, start + k, :] = f

        return X, Z

# --- END OF FILE sandbox/feature_extractor.py ---