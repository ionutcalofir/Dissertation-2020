import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class HighlightDetection():
    def __init__(self, game_path):
        self._game_path = game_path

        self._sliding_windows = json.load(open(os.path.join(game_path, 'sliding_window_videos_information/sliding_window_videos_information.json'), 'r'))
        self._predictions = json.load(open(os.path.join(game_path, 'sliding_window_videos_information/sliding_window_videos_predictions.json'), 'r'))

        actions = [action for action in sorted(os.listdir(os.path.join(game_path, 'gt_videos')))
                        if 'expected_goals' not in action]

        self._actions = {}
        for action in actions:
            action_name = '_'.join(action.split('_')[1:-1])
            frame = action.split('_')[-1][:-4]

            self._actions['frame_{}'.format(frame)] = {}
            self._actions['frame_{}'.format(frame)]['action_name'] = action_name
            self._actions['frame_{}'.format(frame)]['path'] = os.path.join(game_path, 'gt_videos', action)

    def highlight(self):
        predictions_x = []
        predictions_y = []
        predictions_colors = []
        for idx, (sliding_window, prediction) in enumerate(self._predictions.items()):
            start_frame_idx = self._sliding_windows[sliding_window]['start_frame_idx']
            end_frame_idx = self._sliding_windows[sliding_window]['end_frame_idx']
            prediction_idx = np.argmax(prediction)

            if prediction_idx == 0:
                continue

            if start_frame_idx > 1000:
                continue

            predictions_x.append(start_frame_idx)
            predictions_y.append(prediction[prediction_idx])
            predictions_colors.append('b' if prediction_idx == 2 else 'g')

        gt_x = []
        gt_y = []
        gt_colors = []
        for frame in self._actions.keys():
            action_name = self._actions[frame]['action_name']
            frame_idx = int(frame.split('_')[-1])

            if frame_idx > 1000:
                continue

            gt_x.append(frame_idx)
            gt_y.append(1)
            gt_colors.append('b' if action_name == 'shot' else 'g')

        dot_x = np.linspace(min(gt_x) - 100, max(gt_x) + 100).tolist()
        dot_y = [1] * len(dot_x)

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.plot(dot_x, dot_y, color='k', linestyle='dashed')
        ax.scatter(predictions_x, predictions_y, c=predictions_colors)
        ax.scatter(gt_x, gt_y, c=gt_colors)

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Pass',
                                  markerfacecolor='g', markersize=5),
                           Line2D([0], [0], marker='o', color='w', label='Shot',
                                  markerfacecolor='b', markersize=5)]
        ax.legend(handles=legend_elements)
        plt.show()
