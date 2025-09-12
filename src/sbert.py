from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


# node edges list to find clusters of nodes
def cluster_nodes(edges: list[list[int, int]]):
    # Function to find the root of a node
    def find(node, parent):
        if parent[node] == node:
            return node
        else:
            return find(parent[node], parent)

    # Function to union two nodes
    def union(node1, node2, parent):
        root1 = find(node1, parent)
        root2 = find(node2, parent)
        if root1 != root2:
            parent[root2] = root1

    # Find all unique nodes
    nodes = set()
    for edge in edges:
        nodes.update(edge)

    # Initialize parent pointers (each node is its own parent)
    parent = {node: node for node in nodes}

    # Apply union operation for each edge
    for node1, node2 in edges:
        union(node1, node2, parent)

    # Find clusters by grouping nodes with the same root
    clusters = {}
    for node in nodes:
        root = find(node, parent)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(node)

    # Return the list of clusters
    return list(clusters.values())


# returns an index cluster
# https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
def cluster_similar_sentences(
    sentences: list[str], cosine_threshold: float=.7
) -> list[list[int]]:
    embeddings = model.encode(sentences)

    similarities = model.similarity(embeddings, embeddings)
    similarity_edges = np.column_stack(np.where((similarities >= cosine_threshold)))

    clusters = cluster_nodes(similarity_edges.tolist())

    # sort by avg similarity
    sorting_list = [
        -1 for _ in range(similarities.shape[0])
    ]  # -1 to get rid of the duplicate on diag
    for edge in similarity_edges:
        sorting_list[edge[0]] += similarities[edge[0], edge[1]].numpy()

    for cluster in clusters:
        len_cluster = len(cluster)
        if len_cluster == 1:
            continue
        cluster.sort(
            reverse=True, key=lambda x: sorting_list[x] / len_cluster
        )  # sort by avg similarity through nodes

    return clusters  # sorted by avg similarity in cluster
