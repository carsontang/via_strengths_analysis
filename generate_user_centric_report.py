import sys
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20,20]

from analytics import (
    draw_user_centric_graph,
)
from constants import ALL_FILE, BAT_SIGNAL_FILE, TENNCREW_FILE
from data import DataLoader


def generate_report(name):
    user_strengths = DataLoader.load_user_strengths(BAT_SIGNAL_FILE)
    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(300, 300))
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(300, 300))

    # for ax, name in zip(ax, ["Lydia Han", "Ivan Zhou"]):
    #     draw_user_centric_graph(user_strengths, name, ax)
    draw_user_centric_graph(user_strengths, name)
    # ax = plt.gca()
    # ax.set_title("VIA Strengths Analysis")
    filename = "_".join(name.split(" "))
    plt.savefig(filename + ".pdf")
    plt.savefig("couples.pdf")


if __name__ == "__main__":
    generate_report(sys.argv[1])