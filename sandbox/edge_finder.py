import networkx as nx
import numpy as np
from collections import defaultdict
import yaml
from typing import List, Dict, Any, Tuple, Set
from sandbox.utils import get_embedding, cosine_similarity

def normalize_scores_new(scores: Dict[Any, float]) -> Dict[Any, float]:
    """Normalizes scores to a 0-1 range."""
    if not scores:
        return {}
    try:
        max_score = max(scores.values())
        min_score = min(scores.values())
        score_range = max_score - min_score
        if score_range == 0:
            norm_val = 1.0 if max_score > 0 else 0.0
            return {k: norm_val for k in scores}
        else:
            return {k: (v - min_score) / score_range for k, v in scores.items()}
    except Exception:
        return scores

def find_init_topk_edges(
    agent_list: List[Dict[str, Any]],
    top_k: int = 5,
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float]]:
    """
    Identifies the most critical edges in the initial topology using edge betweenness centrality.
    Ensures that source nodes with outgoing edges are represented.
    """
    if top_k <= 0:
        return {}, {}

    G = nx.DiGraph()
    source_nodes_with_edges = set()
    nodes_in_system = set()

    if not agent_list:
        return {}, {}

    for agent_info in agent_list:
        source_id = agent_info.get("id")
        if source_id is None:
            continue

        nodes_in_system.add(source_id)
        G.add_node(source_id) 

        neighbors = agent_info.get("neighborhood", [])
        if neighbors:
            source_nodes_with_edges.add(source_id)

        for target_id in neighbors:
            nodes_in_system.add(target_id)
            if target_id not in G:
                 G.add_node(target_id)
            G.add_edge(source_id, target_id)

    if not G.edges:
        return {}, {}

    try:
        edge_centrality = nx.edge_betweenness_centrality(G, normalized=True)
    except Exception:
        return {}, {}
    
    edge_scores = edge_centrality 

    if not edge_scores:
        return {}, {}

    best_outgoing_edge_for_source: Dict[int, Tuple[Tuple[int, int], float]] = {}
    for (u, v), score in edge_scores.items():
        if u in source_nodes_with_edges:
            if u not in best_outgoing_edge_for_source or score > best_outgoing_edge_for_source[u][1]:
                best_outgoing_edge_for_source[u] = ((u, v), score)

    final_top_edges: Set[Tuple[int, int]] = set()

    guaranteed_candidates = sorted(best_outgoing_edge_for_source.values(), key=lambda item: item[1], reverse=True)

    for edge_tuple, score in guaranteed_candidates:
        if len(final_top_edges) < top_k:
            final_top_edges.add(edge_tuple)
        else:
            break 

    if len(final_top_edges) < top_k:
        all_edges_sorted_by_score = sorted(edge_scores.items(), key=lambda item: item[1], reverse=True)

        for edge_tuple, score in all_edges_sorted_by_score:
            if len(final_top_edges) >= top_k:
                break
            final_top_edges.add(edge_tuple)

    output_dict: Dict[int, List[int]] = defaultdict(list)
    for u, v in final_top_edges:
        output_dict[u].append(v)

    return dict(output_dict), edge_scores

def update_topk_edges(
    goal_embedding: List[str],
    agent_list: List[Dict],
    log: List[Dict],
    initial_static_scores: Dict[Tuple[int, int], float],
    top_k: int,
    client: Any = None  # Changed: Inject client instead of re-initializing
) -> Dict[int, List[int]]:
    """
    Calculates dynamic edge weights based on:
    1. Static centrality (initial importance)
    2. Communication frequency (interaction heat)
    3. Semantic relevance to misinformation goal (risk factor)
    Returns the new optimized topology.
    """
    
    try:
        with open("config/defense.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        weights = config.get("weights", {'static': 0.3, 'frequency': 0.3, 'relevance': 0.4})
        similarity_threshold = config.get("similarity_threshold", 0.7)
    except:
        weights = {'static': 0.3, 'frequency': 0.3, 'relevance': 0.4}
        similarity_threshold = 0.7

    if top_k <= 0:
        return {}
    G = nx.DiGraph()
    all_edges = set()
    if agent_list:
        for agent_info in agent_list:
            source_id = agent_info["id"]
            G.add_node(source_id)
            for target_id in agent_info.get("neighborhood", []):
                 G.add_edge(source_id, target_id)
                 all_edges.add((source_id, target_id))
    
    if not all_edges:
         return {}

    if not goal_embedding:
        weights['relevance'] = 0.0

    edge_frequency: Dict[Tuple[int, int], int] = defaultdict(int)
    edge_max_relevance: Dict[Tuple[int, int], float] = defaultdict(float)

    for entry in log:
        if entry.get("action_type") == "send_message":
            source = entry.get("subjective")
            targets = entry.get("objective", [])
            message_content = entry.get("context", "")

            if source is None or not targets or not message_content:
                continue

            max_sim_message = 0.0
            
            if goal_embedding is not None and weights.get('relevance', 0) > 0 and client:
                try:
                    import nltk
                    try:
                        sentences = nltk.sent_tokenize(message_content)
                    except LookupError:
                        nltk.download('punkt')
                        sentences = nltk.sent_tokenize(message_content)
                except ImportError:
                    # Simple fallback if nltk is missing
                    import re
                    sentences = re.split(r'(?<=[.?!])\s+', message_content)

                for sentence in sentences:
                    if not sentence.strip(): continue
                    sentence_embedding = get_embedding(sentence, client=client) 
                    
                    if sentence_embedding is not None and len(goal_embedding):
                        sim = 0.0
                        for goal in goal_embedding:
                            sim += cosine_similarity(sentence_embedding, goal)
                        sim /= len(goal_embedding) 
                        if sim > similarity_threshold: 
                            max_sim_message = max(max_sim_message, sim)

            for target in targets:
                edge = (source, target)
                if G.has_edge(source, target):
                    edge_frequency[edge] += 1
                    edge_max_relevance[edge] = max(edge_max_relevance.get(edge, 0.0), max_sim_message)

    norm_frequency_scores = normalize_scores_new({e: float(f) for e, f in edge_frequency.items()})
    norm_relevance_scores = edge_max_relevance 

    edge_dynamic_scores: Dict[Tuple[int, int], float] = {}
    
    for u, v in G.edges():
        edge = (u, v)
        s_static = initial_static_scores.get(edge, 0.0) 
        s_freq = norm_frequency_scores.get(edge, 0.0)
        s_rel = norm_relevance_scores.get(edge, 0.0)

        score = (weights.get('static', 0) * s_static +
                 weights.get('frequency', 0) * s_freq +
                 weights.get('relevance', 0) * s_rel)
        edge_dynamic_scores[edge] = score

    sorted_edges = sorted(edge_dynamic_scores.items(), key=lambda item: item[1], reverse=True)
    top_edges_list = [edge for edge, score in sorted_edges[:top_k]]

    output_dict: Dict[int, List[int]] = defaultdict(list)
    for u, v in top_edges_list:
        output_dict[u].append(v)

    return dict(output_dict)