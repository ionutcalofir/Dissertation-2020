import os
import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, cohen_kappa_score

def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):
        """
        Function from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/visualization/utils.py#L48
        """
        if class_names is None or type(class_names) != list:
            class_names = [str(i) for i in range(num_classes)]

        figure = plt.figure(figsize=figsize)
        plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cmtx.max() / 2.0
        for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
            color = "white" if cmtx[i, j] > threshold else "black"
            plt.text(
                j,
                i,
                format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
                horizontalalignment="center",
                color=color,
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        return figure

MATCHES_SHOTS_DIR = 'matches_shots'

leagues = ['epl', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1']
years = ['2014', '2015', '2016', '2017', '2018', '2019']

y_true = []
y_pred = []
y_prob = []
for league in leagues:
    for year in years:
        matches_shots_path = os.path.join(MATCHES_SHOTS_DIR, 'matches_shots_{}_{}.json'.format(league, year))
        matches_shots = json.load(open(matches_shots_path, 'r'))

        for match_shots in matches_shots:
            if match_shots is None:
                continue

            for key in match_shots.keys():
                for shot in match_shots[key]:
                    result = shot['result']
                    xG = float(shot['xG'])

                    if result == 'Goal':
                        y_true.append(1)
                    else:
                        y_true.append(0)

                    if xG >= 0.5:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)

                    y_prob.append(xG)

cm = confusion_matrix(y_true, y_pred)
tp = cm[1, 1]
fp = cm[0, 1]
fn = cm[1, 0]
figure = plot_confusion_matrix(cm, 2, ['no_goal', 'goal'])
plt.show()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * ((precision * recall) / (precision + recall))

print(cm)
print()

print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1: {}'.format(f1))
print()

print('Random AUC-PR: {}'.format(sum(y_true) / len(y_true)))
print('AUC-PR: {}'.format(average_precision_score(y_true, y_prob)))
print('Cohen Kappa: {}'.format(cohen_kappa_score(y_true, y_pred)))
