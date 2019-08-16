import sys
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20,20]

from analytics import (
    generate_user_centric_graph,
)
from constants import ALL_FILE, BAT_SIGNAL_FILE
from data import DataLoader


def generate_report(name):
    user_strengths = DataLoader.load_user_strengths(ALL_FILE)
    generate_user_centric_graph(user_strengths, name)


if __name__ == "__main__":
    generate_report(sys.argv[1])