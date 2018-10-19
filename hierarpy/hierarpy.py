import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


# Including a mini testing dataframe here:
d = pd.DataFrame({'Winner': {0: 'A', 1: 'A', 2: 'B', 3: 'A',
                             4: 'C', 5: 'B', 6: 'C'},
                  'Loser': {0: 'B', 1: 'C', 2: 'A', 3: 'C',
                            4: 'D', 5: 'C', 6: 'D'}})


def matrix_from_dataframe(interaction_df):
    """
    Take an interactions dataframe with 2 columns (Winner, Loser), 
    and turn it into a matrix of interactions
    """
    mat = pd.crosstab(interaction_df['Winner'], interaction_df['Loser'])
    # Make sure the matrix is symmetric
    inds = sorted(set(list(mat)).union(mat.index))
    mat = mat.reindex(inds, fill_value=0)
    mat = mat.reindex(inds, fill_value=0, axis=1)
    return mat


def preprocess(interaction_mat):
    """
    Take matrix of interactions, and subtract bi-directional interactions
    This is useful for ADAGIO
    """
    return (np.clip(interaction_mat-interaction_mat.T,
                    a_min=0, a_max=np.inf)
            .astype(int))


def run_ADAGIO(interaction_matrix, preprocess_data=True, plot=False):
    def plot_graph(graph, edges, labels=False):
        pos = nx.circular_layout(graph)
        plt.cla()
        nx.draw_networkx_nodes(graph, pos=pos)
        nx.draw_networkx_labels(graph, pos=pos)
        nx.draw_networkx_edges(edges, pos=pos)
        if labels:
            nx.draw_networkx_edge_labels(edges, pos=pos)
        plt.axis('off')
    if preprocess_data:
        interaction_matrix = preprocess(interaction_matrix)
    # Transform matrix into useable dataframe for nx:
    d = defaultdict(list)
    for i, col in interaction_matrix.iteritems():
        [d['weight'].append(j) for j in list(col)]
        [d['to'].append(i) for _ in list(col)]
        [d['from'].append(col.index[idx]) for idx in range(len(list(col)))]
    M = pd.DataFrame(d)
    # Make graph
    H = nx.from_pandas_dataframe(M, 'from', 'to', 'weight',
                                 create_using=nx.DiGraph())
    # Select those edges with a weight of > 0
    H_edges = nx.DiGraph(((u, v, e) for u, v, e in H.edges(data=True)
                          if e['weight'] > 0))
    if plot:
        fig = plt.figure(figsize=(11, 7.5))
        fig.add_subplot(121)
        plot_graph(H, H_edges, labels=True)
        plt.title('original graph')
    # pick out the largest scc:
    max_scc = max(nx.strongly_connected_components(H_edges), key=len)
    # Subset the data to only consider the largest scc
    M_scc = M[(M['to'].isin(max_scc)) & (M['from'].isin(max_scc))
              & (M['weight'] > 0)]
    # Get edges of the scc
    H_scc = nx.from_pandas_dataframe(M_scc, 'from', 'to', 'weight',
                                     create_using=nx.DiGraph())
    # number of cyclical groups:
    n_cycl = len(H_scc)
    # Repeat the Process until there are no longer any cyclical groups
    while n_cycl > 0:
        # Find the largest weight in scc:
        idx = M_scc[M_scc['weight'] == min(M_scc['weight'])].index
        # Drop that edge
        M.loc[idx, 'weight'] = 0
        H = nx.from_pandas_dataframe(M, 'from', 'to', 'weight',
                                     create_using=nx.DiGraph())
        H_edges = nx.DiGraph(((u, v, e) for u, v, e in H.edges(data=True)
                              if e['weight'] > 0))
        max_scc = max(nx.strongly_connected_components(H_edges), key=len)
        M_scc = M[(M['to'].isin(max_scc)) & (M['from'].isin(max_scc))
                  & (M['weight'] > 0)]
        H_scc = nx.from_pandas_dataframe(M_scc, 'from', 'to', 'weight',
                                         create_using=nx.DiGraph())
        n_cycl = len(H_scc)
    if plot:
        fig.add_subplot(122)
        plot_graph(H, H_edges, labels=True)
        plt.title('Graph with ADAGIO')
        plt.show()
    return H, H_edges


def rank_from_graph(nodes, edges, method='bottom_up'):
    inds = [i for i in nodes]
    rank_data = {'ind': inds, 'adagio_rank': [0] * len(inds)}
    rank_df = pd.DataFrame(rank_data)
    # Validate all nodes and edges (no need to do it within the graph itself):
    node_val = {}
    edge_val = {}
    for i in nodes.nodes:
        node_val[i] = True
    for i in edges.edges:
        edge_val[i] = True
    # Count valid nodes:
    n_valnodes = sum([1 for _, val in node_val.items() if val])
    # Initiate r
    r = 1
    # Iterate til no more valid nodes
    while n_valnodes > 0:
        # Find valid nodes with no valid edges pointing toward them
        if method == 'bottom_up':
            terminal_nodes = set([node for node in nodes if node_val[
                node]]) - set([edge[0] for edge, val
                               in edge_val.items() if(val & node_val[
                                   edge[1]])])
        elif method == 'top_down':
            terminal_nodes = set([node for node in nodes if node_val[
                node]]) - set([edge[1] for edge, val
                               in edge_val.items() if(val & node_val[
                                   edge[0]])])
        else:
            print("Invalid Method. Please choose top_down or bottom_up")
            return -1
        rank_df.loc[rank_df['ind'].isin(terminal_nodes), 'adagio_rank'] = r
        # Invalidate nodes and edges:
        for node in terminal_nodes:
            node_val[node] = False
        for node in terminal_nodes:
            inv = [edge for edge in edges.edges if edge[0] == node]
            for edge in inv:
                edge_val[edge] = False
        r += 1
        n_valnodes = sum([1 for _, val in node_val.items() if val])
    if method == 'bottom_up':
        rank_df['adagio_rank'] = max(
            rank_df['adagio_rank']) - rank_df['adagio_rank'] + 1
    return rank_df.sort_values('adagio_rank')[['ind', 'adagio_rank']]


def david_ranks(interaction_matrix):
    """
    David's Ranks. 
    Takes as input a matrix of interactions (raw interaction scores)
    Returns Scores, and rankings
    # TODO: Add in linearity measures (see deVries1995)
    # TODO: Check on how to deal with ties
    """
    ids = list(interaction_matrix)
    props = (interaction_matrix/(
        interaction_matrix+interaction_matrix.transpose())).fillna(0.)
    l = pd.Series(props.sum(axis=0), name='l')
    w = pd.Series(props.sum(axis=1), name='w')
    props = props.append(l)
    props = props.assign(w=w)
    w2 = [sum([props.loc[i, j] * props.loc[j, 'w'] for j in ids]) for i in ids]
    w2 = pd.Series(w2, name='w2', index=ids)
    props = pd.concat([props, w2], axis=1)
    l2 = [sum([props.loc[j, i] * props.loc['l', j] for j in ids]) for i in ids]
    l2 = pd.Series(l2, name='l2', index=ids)
    props = props.append(l2)
    props = props.assign(DS=w+w2 - l-l2)
    props = props.sort_values('DS', ascending=False)
    ds = list(props['DS'][:len(ids)])
    ranks = pd.Series(range(1, len(ids)+1))
    david_ranks = pd.DataFrame({'ind': props.index[:len(ids)],
                                'D_score': ds,
                                'Davids_rank': range(
        1, len(ids)+1)}, index=range(1, len(ids)+1))[
            ['ind', 'D_score', 'Davids_rank']]
    return david_ranks


def elo_ranks(interaction_dataframe, k=100):
    """
    Needs some serious refactoring!!
    """
    def elo_logi(diff):
        return(1-(1/(1+10**(diff/400))))
    inds = np.unique(d[['Winner', 'Loser']].values.flatten())
    #
    elo_df = pd.DataFrame(inds, columns=['ind']).assign(
        elo_score=1000).sort_values('ind')
    for i in range(len(interaction_dataframe)):
        row = interaction_dataframe.iloc[i]
        a, b = row['Winner'], row['Loser']
        if (a in inds) & (b in inds):
            a_score = elo_df.loc[elo_df['ind'] == a, 'elo_score'].iloc[0]
            b_score = elo_df.loc[elo_df['ind'] == b, 'elo_score'].iloc[0]
            a_score_old = a_score.copy()
            b_score_old = b_score.copy()
            elo_df.loc[elo_df['ind'] == a, 'elo_score'] = a_score_old + \
                (1-elo_logi(a_score_old-b_score_old)) * k
            elo_df.loc[elo_df['ind'] == b, 'elo_score'] = b_score_old + \
                (0-elo_logi(b_score_old-a_score_old)) * k
    elo_df = elo_df.sort_values('elo_score', ascending=False).assign(
        elo_rank=range(1, len(elo_df)+1)).reset_index(drop=True)
    return elo_df

# TODO: Randomized Elo
# TODO: Serious refactoring of elo_ranks, probably also of david_ranks
# TODO: Linearity Scores
