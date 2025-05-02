import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import networkx as nx


def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=1, topk_e=1, cost_e=0.5):
    print("retrieval_via_pcst")
    g= nx.Graph()
    g.add_nodes_from([i for i in range(graph.x.shape[0])])
    edge_index_ = np.array((graph.edge_index))
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in
                            range(np.shape(edge_index_)[1])]
    
    g.add_edges_from(edge_index)

    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)
        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    #! Extract nodes from whole graph based on similarity
    print(f"topk {topk} nodeds: {topk_n_indices}")
    v_labels = []
    for node_indice in topk_n_indices:
        temp_list = [name for name, value in nx.single_source_shortest_path_length(g, int(node_indice), cutoff=1).items()]
        v_labels.extend(temp_list)

    #! Extract related edges based on extracted nodes
    print("Extract related edges based on extracted nodes")
    x = graph.x[v_labels]
    Subgraph = g.subgraph(v_labels)
    all_edges = list(tuple(e) for e in g.edges)
    sub_edges = list(tuple(e) for e in Subgraph.edges)

    edge_indices = []
    for sub_edge in sub_edges:
        index = next((i for i, x in enumerate(edge_index) if x == sub_edge), -1)
        if index != -1:
            edge_indices.append(index)
        else:
            rebuild_edges = [sub_edge[1], sub_edge[0]]
            index = next((i for i, x in enumerate(edge_index) if x == rebuild_edges), -1)
            if index != -1:
                edge_indices.append(index)

    n = textual_nodes.iloc[v_labels]
    # print(f"nodes list: {n}")
    e = textual_edges.iloc[edge_indices]
    # print(f"edges list: {e}")

    # ! Only return the Node list, if edge information need to be provied, developer can add edge information here
    desc = "There are the mappings of node_id to node decription:\n" + n.to_csv(index=False)+'\n'
    edge_attr = graph.edge_attr[edge_indices]
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(v_labels))

    return data, desc
