"""Compare testing performance data"""
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from matplotlib.lines import Line2D


# Data

data = {
    "Akshay": {
        "mode": "Convoy",
        "sim":  {"short": [85.24, 79.98], "long": [92.04, 89.80]},
        "real": {"short": [23.87, 21.32], "long": [31.47, 25.52]},
    },
    "Lorena": {
        "mode": "Teleop",
        "sim":  {"short": [262.58, 168.17], "long": [224.50, 223.83]},
        "real": {"short": [81.38, 60.00],   "long": [55.57,  61.85]},
    },
    "Rio": {
        "mode": "Convoy",
        "sim":  {"short": [62.58, 64.28],   "long": [108.61, 82.18]},
        "real": {"short": [40.84, 57.10],   "long": [33.13,  46.45]},
    },
    "Sarvesh": {
        "mode": "Teleop",
        "sim":  {"short": [172.62, 163.52], "long": [216.86, 234.51]},
        "real": {"short": [36.40,  46.69],  "long": [53.00,  41.73]},
    },
    "Drake": {
        "mode": "Convoy",
        "sim":  {"short": [65.48, 59.67],   "long": [93.95,  94.84]},
        "real": {"short": [46.13, 20.85],   "long": [27.45,  21.61]},
    },
    "Jacky": {
        "mode": "Teleop",
        "sim":  {"short": [424.86, 306.36], "long": [384.46, 278.85]},
        "real": {"short": [102.98, 63.51],  "long": [100.43, 166.40]},
    },
    "Alyssa": {
        "mode": "Convoy",
        "sim":  {"short": [59.97, 57.50],   "long": [77.07,  78.02]},
        "real": {"short": [17.95, 16.70],   "long": [22.56,  16.20]},
    },
    "Ananya": {
        "mode": "Teleop",
        "sim":  {"short": [504.83, 493.53], "long": [520.92, 333.80]},
        "real": {"short": [91.55,  72.04],  "long": [125.65, 71.32]},
    },
    "Damla": {
        "mode": "Convoy",
        "sim":  {"short": [252.79, 190.49], "long": [279.21, 207.30]},
        "real": {"short": [182.23, 105.45], "long": [106.25, 65.58]},
    },
    "Aidan": {
        "mode": "Teleop",
        "sim":  {"short": [336.32, 217.02], "long": [263.28, 231.94]},
        "real": {"short": [29.60,  26.33],  "long": [48.12,  37.57]},
    },
    "Austin": {
        "mode": "Convoy",
        "sim":  {"short": [83.03,  92.11],  "long": [90.23,  87.46]},
        "real": {"short": [14.20,  59.32],  "long": [38.87,  31.15]},
    },
}



# Config
plot_num = 0

# Box Plot with Markers
pt_color = "blue"
shift = 0.2
wiggle = 0.015

# Box Plot with Markers by Speed
fast_color = "#38EE93"
med_color = "#F1E04B"
slow_color = "#EC7E18"





# Helpers

def best_attempts(data):
    result = {}
    for participant, d in data.items():
        result[participant] = {
            "mode": d["mode"],
            "sim":  {"short": min(d["sim"]["short"]), "long": min(d["sim"]["long"])},
            "real":   {"short": min(d["real"]["short"]),  "long": min(d["real"]["long"])},
        }
    return result



# Plot

bests = best_attempts(data)

data_lists = {
    "sim_short_convoy": [],
    "sim_long_convoy": [],
    "real_short_convoy": [],
    "real_long_convoy": [],
    "sim_short_teleop": [],
    "sim_long_teleop": [],
    "real_short_teleop": [],
    "real_long_teleop": [],
}

for participant, data in bests.items():
    mode = data["mode"]

    if mode == "Convoy":
        data_lists["sim_short_convoy"].append(data["sim"]["short"])
        data_lists["sim_long_convoy"].append(data["sim"]["long"])
        data_lists["real_short_convoy"].append(data["real"]["short"])
        data_lists["real_long_convoy"].append(data["real"]["long"])
    else:
        data_lists["sim_short_teleop"].append(data["sim"]["short"])
        data_lists["sim_long_teleop"].append(data["sim"]["long"])
        data_lists["real_short_teleop"].append(data["real"]["short"])
        data_lists["real_long_teleop"].append(data["real"]["long"])




# Fig 1
# fig, axs = plt.subplots(2,2, figsize=(8,8))

# conditions = [
#     ("sim",  "short", axs[0,0], "Sim: Short"),
#     ("sim",  "long",  axs[0,1], "Sim: Long"),
#     ("real", "short", axs[1,0], "Real: Short"),
#     ("real", "long",  axs[1,1], "Real: Long"),
# ]

# for env, course, ax, title in conditions:
#     convoy = data_lists[f"{env}_{course}_convoy"]
#     teleop = data_lists[f"{env}_{course}_teleop"]

#     ax.boxplot([convoy, teleop], tick_labels=["Convoy", "Teleop"])
#     # ax.scatter(1*np.ones_like(convoy)+max_shift*(2*np.random.random(len(convoy))-1), convoy, zorder=3, color=pt_color, s=20)
#     # ax.scatter(2*np.ones_like(teleop)+max_shift*(2*np.random.random(len(teleop))-1), teleop, zorder=3, color=pt_color, s=20)
#     ax.scatter(1*np.ones_like(convoy)+shift+wiggle*(-1)**np.arange(len(convoy)), np.sort(convoy), zorder=3, color=pt_color, s=20)
#     ax.scatter(2*np.ones_like(teleop)-shift+wiggle*(-1)**np.arange(len(teleop)), np.sort(teleop), zorder=3, color=pt_color, s=20)
#     ax.set_title(title)
#     ax.set_ylabel("Completion Time (s)")


# fig.tight_layout()




# Fig 2
fig, axs = plt.subplots(2,2, figsize=(6,6), sharex="col", sharey="row")

conditions = [
    ("sim",  "short", axs[0,0], "Sim: Short"),
    ("sim",  "long",  axs[0,1], "Sim: Long"),
    ("real", "short", axs[1,0], "Real: Short"),
    ("real", "long",  axs[1,1], "Real: Long"),
]

first = True
third = False
for env, course, ax, title in conditions:
    convoy = data_lists[f"{env}_{course}_convoy"]
    teleop = data_lists[f"{env}_{course}_teleop"]
    
    ax.boxplot([convoy, teleop], tick_labels=["Convoy", "Teleop"])

    # ax.boxplot([np.sort(convoy)[0:math.ceil(len(convoy)/3)], np.sort(teleop)[0:math.ceil(len(teleop)/3)]], tick_labels=["Convoy", "Teleop"])
    ax.scatter(1*np.ones_like(convoy)[0:math.ceil(len(convoy)/3)]+shift+wiggle*(-1)**np.arange(len(convoy))[0:math.ceil(len(convoy)/3)], np.sort(convoy)[0:math.ceil(len(convoy)/3)], zorder=5, color=fast_color, s=20)
    ax.scatter(2*np.ones_like(teleop)[0:math.ceil(len(teleop)/3)]-shift+wiggle*(-1)**np.arange(len(teleop))[0:math.ceil(len(teleop)/3)], np.sort(teleop)[0:math.ceil(len(teleop)/3)], zorder=5, color=fast_color, s=20)

    # ax.boxplot([np.sort(convoy)[math.ceil(len(convoy)/3):2*math.ceil(len(convoy)/3)], np.sort(teleop)[math.ceil(len(teleop)/3):2*math.ceil(len(teleop)/3)]], tick_labels=["Convoy", "Teleop"])
    ax.scatter(1*np.ones_like(convoy)[math.ceil(len(convoy)/3):2*math.ceil(len(convoy)/3)]+shift+wiggle*(-1)**np.arange(len(convoy))[math.ceil(len(convoy)/3):2*math.ceil(len(convoy)/3)], np.sort(convoy)[math.ceil(len(convoy)/3):2*math.ceil(len(convoy)/3)], zorder=4, color=med_color, s=20)
    ax.scatter(2*np.ones_like(teleop)[math.ceil(len(teleop)/3):2*math.ceil(len(teleop)/3)]-shift+wiggle*(-1)**np.arange(len(teleop))[math.ceil(len(teleop)/3):2*math.ceil(len(teleop)/3)], np.sort(teleop)[math.ceil(len(teleop)/3):2*math.ceil(len(teleop)/3)], zorder=4, color=med_color, s=20)

    # ax.boxplot([np.sort(convoy)[2*math.ceil(len(convoy)/3):], np.sort(teleop)[2*math.ceil(len(teleop)/3):]], tick_labels=["Convoy", "Teleop"])
    ax.scatter(1*np.ones_like(convoy)[2*math.ceil(len(convoy)/3):]+shift+wiggle*(-1)**np.arange(len(convoy))[2*math.ceil(len(convoy)/3):], np.sort(convoy)[2*math.ceil(len(convoy)/3):], zorder=3, color=slow_color, s=20)
    ax.scatter(2*np.ones_like(teleop)[2*math.ceil(len(teleop)/3):]-shift+wiggle*(-1)**np.arange(len(teleop))[2*math.ceil(len(teleop)/3):], np.sort(teleop)[2*math.ceil(len(teleop)/3):], zorder=3, color=slow_color, s=20)

    ax.set_title(title)

    

    if first or third:
        ax.set_ylabel("Completion Time (s)")

    if not first and not third:
        third = True
    elif third:
        third = False

    if first:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=slow_color,  markersize=6, label='Lower Third'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=med_color,  markersize=6, label='Middle Third'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=fast_color, markersize=6, label='Top Third'),
        ]
        ax.legend(handles=legend_elements)
        first = False


fig.tight_layout()


plt.show()



