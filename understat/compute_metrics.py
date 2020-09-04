import os
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, cohen_kappa_score

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
