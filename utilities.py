import numpy as np
import matplotlib.pyplot as plt


def plot_attribute(avg_attr):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.array(avg_attr)
    y = np.sort(x)

    plt.title("Average Aggression in Population")
    plt.plot(y, color="red")

    plt.show()


