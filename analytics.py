import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from constants import (
    DRAW_LABELS,
)


def shared_top3_adj(user_strengths):
    users = list(user_strengths.keys())

    d = {}
    index = {}
    users.sort()  # Needed so that we can create a diagonal matrix
    for i, user in enumerate(users):
        d[user] = [0 for j in range(len(users))]
        index[i] = user

    df = pd.DataFrame(d, dtype=np.float64)
    df = df.rename(index=index)

    for user in users:
        for other_user in users:
            top3 = set(user_strengths[user][:3])
            other_top3 = set(user_strengths[other_user][:3])
            shared_top3 = (top3 & other_top3)
            if shared_top3:
                df[user][other_user] = 1
                df[other_user][user] = 1
            else:
                df[user][other_user] = 0
                df[other_user][user] = 0
    return df


def generate_via_strengths_graph(user_strengths):
    G = nx.Graph()
    users = list(user_strengths.keys())
    node_sizes = []

    # Populate graph with users. Nodes are identified by user names.
    for name in users:
        G.add_node(name)
        node_sizes.append(400)

    # Connect two users if they share a top 3 strength
    for i, user in enumerate(users):
        for j, other_user in enumerate(users[i + 1:]):
            top3 = set(user_strengths[user][:3])
            other_top3 = set(user_strengths[other_user][:3])
            shared_top3 = (top3 & other_top3)

            if shared_top3:
                G.add_edge(user, other_user)

    adj_sizes = []
    for i, n in enumerate(G.nodes()):
        adj_list_size = len(G.adj[n])
        node_sizes[i] *= adj_list_size
        adj_sizes.append(adj_list_size)

    adj_sizes = np.array(adj_sizes)

    mean = adj_sizes.mean()
    std = adj_sizes.std()

    lower_band = mean - std
    upper_band = mean + std

    node_colors = []
    for i, n in enumerate(G.nodes()):
        size = adj_sizes[i]
        if size < lower_band:
            node_colors.append("lime")
        elif size > upper_band:
            node_colors.append("cyan")
        else:
            node_colors.append("lightgreen")

    def create_user_vec(single_user_strengths):
        user_vec = np.empty(24)
        for i, strength in enumerate(single_user_strengths):
            user_vec[strength.value - 1] = i
        return user_vec

    users = list(user_strengths.keys())

    d = {}
    index = {}
    users.sort()  # Needed so that we can create a diagonal matrix
    for i, user in enumerate(users):
        d[user] = [0 for j in range(len(users))]
        index[i] = user

    df = pd.DataFrame(d, dtype=np.float64)
    df = df.rename(index=index)

    for i, user in enumerate(users):
        for j, other_user in enumerate(users):
            if user == other_user:
                df[user][other_user] = 0
                continue
            user_vec = create_user_vec(user_strengths[user])
            other_user_vec = create_user_vec(user_strengths[other_user])
            corrmatrix = np.corrcoef(user_vec, other_user_vec)
            coefficient = corrmatrix[0][1]
            df[user][other_user] = coefficient

    edge_colors = []
    edge_widths = []

    def are_mutual_best_friends(df, u1, u2):
        u1_best_friend = df[u1].idxmax()
        u2_best_friend = df[u2].idxmax()
        return u1_best_friend == u2 and u1 == u2_best_friend

    for i, edge in enumerate(G.edges()):
        u1, u2 = edge

        coef12 = df[u1][u2]

        if are_mutual_best_friends(df, u1, u2):
            edge_colors.append("gold")
            edge_widths.append(6)
        elif coef12 < -0.5:
            edge_colors.append("red")
            edge_widths.append(6)
        elif coef12 < -0.25:
            edge_colors.append("orangered")
            edge_widths.append(4)
        elif coef12 < 0:
            edge_colors.append("salmon")
            edge_widths.append(1)
        elif coef12 > 0.5:
            edge_colors.append("springgreen")
            edge_widths.append(4)
        elif coef12 > 0.25:
            edge_colors.append("lightgreen")
            edge_widths.append(2)
        else:
            edge_colors.append("aquamarine")
            edge_widths.append(1)

    pos = nx.circular_layout(G)

    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, edge_color=edge_colors,
            width=edge_widths, font_weight='bold')
    nx.draw_networkx_labels(G, pos, with_labels=True, font_color='black', node_size=node_sizes)
    plt.savefig('graph.png', figsize=(200, 200))


def create_user_vec(single_user_strengths):
    user_vec = np.empty(24)
    for i, strength in enumerate(single_user_strengths):
        user_vec[strength.value - 1] = i
    return user_vec


def user_coefficients(user_strengths):
    users = list(user_strengths.keys())

    d = {}
    index = {}
    users.sort()  # Needed so that we can create a diagonal matrix
    for i, user in enumerate(users):
        d[user] = [0 for j in range(len(users))]
        index[i] = user

    df = pd.DataFrame(d, dtype=np.float64)
    df = df.rename(index=index)

    for i, user in enumerate(users):
        for j, other_user in enumerate(users):
            if user == other_user:
                df[user][other_user] = 0
                continue
            user_vec = create_user_vec(user_strengths[user])
            other_user_vec = create_user_vec(user_strengths[other_user])
            corrmatrix = np.corrcoef(user_vec, other_user_vec)
            coefficient = corrmatrix[0][1]
            df[user][other_user] = coefficient
    return df


def ccw_dec_coef_layout(G, vip, user_coefficients):
    """
    Users are positioned around a circle. From 0 radians to 2 * np.pi radians,
    in counterclockwise motion (ccw), users of decreasing coefficient are positioned.
    The VIP (very important person) is in the center.
    """
    pos = nx.circular_layout(G)
    num_nodes = len(user_coefficients.index.values) - 1
    rad_delta = 2 * np.pi/num_nodes
    rad = 0
    name_coef = list(user_coefficients[vip].to_dict().items())
    name_coef.sort(key=lambda t: t[1], reverse=True)
    for t in name_coef:
        name, coef = t
        if name == vip:
            continue
        x = np.cos(rad)
        y = np.sin(rad)

        rad += rad_delta
        pos[name] = np.array([x, y])

    pos[vip] = np.array([0.0, 0.0])
    return pos


def draw_user_centric_graph(user_strengths, vip, ax=None):
    if not ax:
        ax = plt.gca()

    G = nx.Graph()
    users = list(user_strengths.keys())
    for name in users:
        G.add_node(name)
        if name == vip:
            continue

    df = user_coefficients(user_strengths)

    edge_colors = []
    edge_widths = []

    # Add edges
    for user in users:
        if vip == user:
            continue
        G.add_edge(vip, user)

    for edge in G.edges():
        u1, u2 = edge

        coef12 = df[u1][u2]
        edge_colors.append(coef12)
        edge_widths.append(10)

    pos = ccw_dec_coef_layout(G, vip, df)

    cmap = plt.cm.get_cmap("plasma")
    colors = []

    for n in G.nodes:
        if vip == n:
            colors.append(1.0)
            continue
        colors.append(df[vip][n])

    sizes = [8000 if n == vip else 800 for n in G.nodes]

    nx.draw(
        G, pos,
        with_labels=False,
        node_color=colors,
        node_size=sizes,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        edge_color=edge_colors,
        edge_vmin=-1.0,
        edge_vmax=1.0,
        width=edge_widths,
        font_weight='bold',
        edge_cmap=cmap,
        ax=ax,
    )

    for name in pos:
        if name == vip:
            continue
        pos[name][1] += 0.05

    pos[vip] = np.array([0.0, 0.0])

    nx.draw_networkx_labels(
        G, pos,
        with_labels=True,
        font_color='black',
        ax=ax
    )
    print("here 2")
    G.clear()
    #
    # filename = "_".join(vip.split(" "))
    # plt.savefig(filename + ".pdf", figsize=(300, 300))