import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from analytics import (
    ccw_dec_coef_layout,
    user_coefficients,
)
from constants import (
    BAT_SIGNAL_FILE,
    DRAW_LABELS,
    EDGE_WIDTH,
    VIP_OUTBOUND_EDGE_WIDTH,
    SET_WIDTH,
)
from data import DataLoader


def calculate_edge_width(G, vip):
    edge_widths = []

    for edge in G.edges():
        u1, u2 = edge
        if u1 == vip or u2 == vip:
            edge_widths.append(VIP_OUTBOUND_EDGE_WIDTH)
        else:
            edge_widths.append(EDGE_WIDTH)
    return edge_widths


def draw_report():
    user_strengths = DataLoader.load_user_strengths(BAT_SIGNAL_FILE)
    df = user_coefficients(user_strengths)
    nrows = 3
    ncols = 2
    figrows = nrows * 20
    figcols = ncols * 20
    fig, axes = plt.subplots(nrows, ncols, figsize=(figcols, figrows))

    names = [
        "Lydia Han",
        "Anhang Zhu",
        "Sophia Han",
        "Ivan Zhou",
        "Lanssie Ma",
        "Edward Wu",
    ]

    def add_edges(graph, vip, users):
        for u in users:
            if u == vip:
                continue
            graph.add_edge(vip, u)

    users = df.index.values
    cmap = plt.cm.get_cmap("plasma")

    for vip, ax in zip(names, axes.flatten()):
        G = nx.Graph()
        for name in users:
            G.add_node(name)
            add_edges(G, name, users)

        node_colors = [df[vip][name] if name != vip else 1.0 for name in G.nodes]
        edge_colors = [df[edge[0]][edge[1]] for edge in G.edges()]

        kwargs = {}

        if SET_WIDTH:
            kwargs["width"] = calculate_edge_width(G, vip)

        pos = ccw_dec_coef_layout(G, vip, df)

        nx.draw(
            G, pos,
            ax=ax,
            with_labels=False,
            cmap=cmap,
            node_color=node_colors,
            vmin=-1.0,
            vmax=1.0,
            edge_cmap=cmap,
            edge_color=edge_colors,
            edge_vmin=-1.0,
            edge_vmax=1.0,
            **kwargs,
        )

        if DRAW_LABELS:
            for name in pos:
                if name == vip:
                    continue
                x, y = pos[name][0], pos[name][1]

                # Use arctan2 instead of arctan.
                # Theory here: https://stackoverflow.com/a/12011762
                rad = np.arctan2(y, x)
                radius = np.sqrt(x**2 + y**2) * 1.2

                new_x, new_y = radius * np.cos(rad), radius * np.sin(rad)
                pos[name] = np.array([new_x, new_y])

            nx.draw_networkx_labels(
                G, pos,
                with_labels=True,
                font_color='black',
                ax=ax
            )
        G.clear()

    fig.tight_layout()

    plt.savefig("dummy.pdf")


if __name__ == "__main__":
    draw_report()