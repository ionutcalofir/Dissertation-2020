import os
import json
import numpy as np
import matplotlib.pyplot as plt

root_path = '/mnt/storage1/dissertation_dataset/games'
test_path = '/mnt/storage1/dissertation_dataset/games/configs/test.txt'

all_xG = []
all_aG = []
with open(test_path, 'r') as f:
    for game in f:
        game_path = os.path.join(root_path, game.strip())

        info = json.load(open(os.path.join(game_path, 'gt_videos_observations/ground_truth_videos_information.json'), 'r'))
        pred = json.load(open(os.path.join(game_path, 'gt_videos_observations/sliding_window_videos_predictions.json'), 'r'))

        xG = 0
        aG = 0
        for key in pred.keys():
            frame_idx = key.split('_')[-1]
            frame_idx = str(info['expected_goals'][frame_idx]['end_frame_idx'] - 2)
            val = info['shot'][frame_idx]

            xG += pred[key][1]
            if val['is_goal']:
                aG += 1

        all_xG.append(xG)
        all_aG.append(aG)

xs = [i for i in range(1, len(all_xG) + 1)]

fig, ax = plt.subplots()
ax.plot(xs, all_xG, label='Expected Goals')
ax.plot(xs, all_aG, label='Actual Goals')
ax.set_title('Expected Goals vs Actual Goals')
plt.legend()


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

labels = ['Actual Goals vs xG']
goals = [sum(all_aG)]
xG = [round(sum(all_xG), 4)]

x = np.arange(len(goals))
width = 0.5

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, goals, width, label='Actual Goals')
rects2 = ax.bar(x + width / 2, xG, width, label='xG')

ax.set_ylabel('Value')
ax.set_title('Actual Goals vs xG')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(0, 75)
ax.legend()

autolabel(ax, rects1)
autolabel(ax, rects2)

fig.tight_layout()
plt.show()
