import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


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


def generate_user_centric_graph(user_strengths, vip):
    G = nx.Graph()
    users = list(user_strengths.keys())
    node_sizes = []
    node_colors = []
    for name in users:
        G.add_node(name)
        if name == vip:
            node_sizes.append(4000)
            node_colors.append("lightgoldenrodyellow")
            continue
        node_sizes.append(800)
        node_colors.append("antiquewhite")

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

    # Add edges
    user = vip
    for i, other_user in enumerate(users):
        if user == other_user:
            continue
        G.add_edge(user, other_user)

    for i, edge in enumerate(G.edges()):
        u1, u2 = edge

        coef12 = df[u1][u2]
        if coef12 < -0.5:
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
            edge_widths.append(6)
        elif coef12 > 0.25:
            edge_colors.append("lightgreen")
            edge_widths.append(2)
        else:
            edge_colors.append("aquamarine")
            edge_widths.append(1)

    pos = nx.circular_layout(G)
    pos[vip] = np.array([0.0, 0.0])
    colors = ["lightgoldenrodyellow" if n == vip else "antiquewhite" for n in G.nodes]
    sizes = [4000 if n == vip else 800 for n in G.nodes]
    nx.draw(G, pos, with_labels=False, node_color=colors, node_size=sizes, edge_color=edge_colors,
            width=edge_widths, font_weight='bold')

    for p in pos:  # raise text positions
        pos[p][1] += 0.04
    pos[vip] = np.array([0.0, 0.0])
    nx.draw_networkx_labels(G, pos, with_labels=True, font_color='black')
    filename = "_".join(vip.split(" "))
    plt.savefig(filename + ".png", figsize=(200, 200))