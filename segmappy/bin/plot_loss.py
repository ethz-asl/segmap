import numpy as np
import os
import json
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder")
args = parser.parse_args()

with open(os.path.join(args.folder, "loss.json"), "r") as fp:
    debug = json.load(fp)
    epoch = debug["epoch"]
    train_loss = debug["train_loss"]
    train_loss_c = debug["train_loss_c"]
    train_loss_r = debug["train_loss_r"]
    train_accuracy = debug["train_accuracy"]
    test_loss = debug["test_loss"]
    test_loss_c = debug["test_loss_c"]
    test_loss_r = debug["test_loss_r"]
    test_accuracy = debug["test_accuracy"]

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

fig_width_pt = 252  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
# golden_mean = (np.sqrt(5)-1.0)/2.0 # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * 0.5  # height in inches
fig_size = [fig_width, fig_height]

fig_loss = plt.figure(figsize=(fig_width, fig_height))

fontsize = 9
params = {
    "backend": "ps",
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    #   'title.fontsize': fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    "figure.figsize": fig_size,
}
plt.rcParams.update(params)
lw = 0.7

train_loss_r = np.array(train_loss_r) * 200
test_loss_r = np.array(test_loss_r) * 200

fig = plt.figure(1)

ax = fig.add_subplot(121, aspect=256.0 / 6.0)
plt.plot(epoch, train_loss_c, lw=lw, color="b", linestyle="-", label="Train")
plt.plot(epoch, test_loss_c, lw=lw, color="g", linestyle="-", label="Test")

plt.xlim([0, 256])
plt.ylim([2, 8])
plt.xticks([0, 128, 256])
plt.yticks([2, 4, 6, 8])

ax.minorticks_on()
ax.xaxis.set_minor_locator(MultipleLocator(64))
ax.yaxis.set_minor_locator(MultipleLocator(1))

plt.grid(b=True, which="major", linestyle="--", lw=0.5)
plt.grid(b=True, which="minor", linestyle="--", lw=0.5)

plt.xlabel("Epoch [\#]")
plt.ylabel("$L_c$ [-]")

plt.legend(loc="upper right")

ax = fig.add_subplot(122, aspect=256.0 / 2.0)
plt.plot(epoch, train_loss_r, lw=lw, color="b", linestyle="-", label="Train")
plt.plot(epoch, test_loss_r, lw=lw, color="g", linestyle="-", label="Test")

plt.xlim([0, 256])
plt.ylim([3, 5])
plt.xticks([0, 128, 256])
plt.yticks([3, 4, 5])

ax.minorticks_on()
ax.xaxis.set_minor_locator(MultipleLocator(64))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))

plt.grid(b=True, which="major", linestyle="--", lw=0.5)
plt.grid(b=True, which="minor", linestyle="--", lw=0.5)

plt.xlabel("Epoch [\#]")
plt.ylabel("$L_r$ [-]")

plt.legend(loc="upper right")
plt.tight_layout()

fig_loss.savefig("loss.png")
fig_loss.savefig("loss.pdf")
