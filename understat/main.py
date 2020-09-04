import json
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = 'matches'

def autolabel(ax, rects):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def make_plot(goals, xG):
    labels = ['Goals vs xG']
    goals = [goals]
    xG = [round(xG, 4)]

    x = np.arange(len(goals))
    width = 0.5

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, goals, width, label='Goals')
    rects2 = ax.bar(x + width / 2, xG, width, label='xG')

    ax.set_ylabel('Values')
    ax.set_title('Goals vs xG')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-1., 1.)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)

    fig.tight_layout()
    plt.show()

def main():
    fixtures = sorted(os.listdir(ROOT_DIR))

    goals = 0
    xG = 0

    for fixture in fixtures:
        matches = json.load(open(os.path.join(ROOT_DIR, fixture)))

        for match in matches:
            goals += float(match['goals']['h'])
            goals += float(match['goals']['a'])
            xG += float(match['xG']['h'])
            xG += float(match['xG']['a'])

    make_plot(goals, xG)

if __name__ == '__main__':
    main()
