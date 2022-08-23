import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def collect_data(self):
    attr_agg_total = 0
    max_agg_attr = 0
    attr_power_total = 0
    attr_cooperation_total = 0
    for particle in self.particles:
        attr_agg_total += particle.aggressiveness
        attr_power_total += particle.power
        attr_cooperation_total += particle.has_cooperated
        if particle.aggressiveness > max_agg_attr:
            max_agg_attr = particle.aggressiveness

    self.avg_agg_attribute.append("%.1f" % (attr_agg_total / len(self.particles)))
    self.avg_power_attribute.append("%.1f" % (attr_power_total / len(self.particles)))
    self.avg_cooperation_attribute.append("%.1f" % (attr_cooperation_total / len(self.particles)))
    self.max_attribute.append(max_agg_attr)


def plot_agg_attribute(avg_attr):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.array(avg_attr)
    y = np.sort(x)

    plt.title("Average Aggression in Population")
    plt.plot(y, color="red")

    plt.show()


def plot_power_attribute(avg_attr):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.array(avg_attr)
    y = np.sort(x)

    plt.title("Average Power in Population")
    plt.plot(y, color="red")

    plt.show()


def plot_cooperation_attribute(avg_attr):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.array(avg_attr)
    y = np.sort(x)

    plt.title("Average Cooperation in Population")
    plt.plot(y, color="red")

    plt.show()

