import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from PIL import Image

class HighlightDetection():
    def __init__(self, root_path, window_length, phase, games_txt):
        self._root_path = root_path
        self._window_length = window_length
        self._phase = phase
        self._games_txt = games_txt

        # 1 - pass
        # 2 - shot
        # These values are computed on a validation set and used for the test set.
        self._w_class = {1: 15,
                         2: 10}
        self._th_class = {1: 0.90,
                          2: 0.90}

    def _get_videos_information(self, game_name, window_length=None):
        if window_length is None:
            window_length = self._window_length

        sliding_windows = json.load(open(os.path.join(self._root_path, game_name,
                                                      'sliding_window_videos/sliding_window_videos_information_length_{}/sliding_window_videos_information.json'.format(window_length)), 'r'))
        predictions = json.load(open(os.path.join(self._root_path, game_name,
                                                  'sliding_window_videos/sliding_window_videos_information_length_{}/sliding_window_videos_predictions.json'.format(window_length)), 'r'))

        all_actions = [action for action in sorted(os.listdir(os.path.join(self._root_path, game_name, 'gt_videos')))
                        if 'expected_goals' not in action]

        actions = {}
        for action in all_actions:
            action_name = '_'.join(action.split('_')[1:-1])
            frame = action.split('_')[-1][:-4]

            actions['frame_{}'.format(frame)] = {}
            actions['frame_{}'.format(frame)]['action_name'] = action_name
            actions['frame_{}'.format(frame)]['path'] = os.path.join(self._root_path, game_name, 'gt_videos', action)

        return actions, sliding_windows, predictions

    def _show_plot(self, gts, predictions):
        gts_x = [gt[0] for gt in gts]
        gts_y = [1] * len(gts_x)
        gts_color = ['b' if gt[1] == 2 else 'g' for gt in gts]

        preds_x = [pred[0] for pred in predictions]
        preds_y = [pred[2] for pred in predictions]
        preds_color = ['b' if pred[1] == 2 else 'g' for pred in predictions]

        dot_x = np.linspace(min(gts_x) - 100, max(gts_x) + 100).tolist()
        dot_y = [1] * len(dot_x)

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.plot(dot_x, dot_y, color='k', linestyle='dashed')
        ax.scatter(preds_x, preds_y, c=preds_color)
        ax.scatter(gts_x, gts_y, c=gts_color)

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Pass',
                                  markerfacecolor='g', markersize=5),
                           Line2D([0], [0], marker='o', color='w', label='Shot',
                                  markerfacecolor='b', markersize=5)]
        ax.legend(handles=legend_elements)
        plt.show()

    def _apply_nms_class(self, gts, predictions, class_name, w_class, th_class):
        # Apply threshold
        predictions = [pred for pred in predictions if pred[2] >= th_class]

        # NMS
        predictions_nms = []
        predictions_prob = np.array([pred[2] for pred in predictions])

        mask = np.zeros(predictions_prob.shape[0], dtype=np.int64)
        while True:
            if np.sum(mask) == len(mask):
                break

            predictions_prob[np.where(mask == 1)] = 0.
            max_prob_idx = np.argmax(predictions_prob)

            predictions_nms.append(predictions[max_prob_idx])
            add_to_mask = [max_prob_idx]

            for idx in range (max_prob_idx - 1, -1, -1):
                if predictions[max_prob_idx][0] - predictions[idx][0] > w_class:
                    break
                add_to_mask.append(idx)

            for idx in range(max_prob_idx + 1, len(predictions)):
                if predictions[idx][0] - predictions[max_prob_idx][0] > w_class:
                    break
                add_to_mask.append(idx)

            add_to_mask = np.array(add_to_mask)
            mask[add_to_mask] = 1

        return predictions_nms

    def _apply_nms(self, gts, predictions, w_class=None, th_class=None):
        if w_class is None and th_class is None: # testing case
            w_class = self._w_class
            th_class = self._th_class

        predictions_nms = []
        for class_name in [1, 2]:
            gts_class = [gt for gt in gts if gt[1] == class_name]
            predictions_class = [pred for pred in predictions if pred[1] == class_name]

            w_class_value = w_class[class_name]
            th_class_value = th_class[class_name]

            predictions_class_nms = self._apply_nms_class(gts_class, predictions_class, class_name, w_class_value, th_class_value)
            predictions_nms.extend(predictions_class_nms)

        predictions_nms.sort(key=lambda x: x[0])
        return predictions_nms

    def _compute_predictions(self, game_name, window_length=None):
        info_actions, info_sliding_windows, info_predictions = self._get_videos_information(game_name, window_length)

        predictions = [] # tuple (start_frame_idx, prediction_idx, probability_prediction_idx, sliding_window)
        for idx, (sliding_window, prediction) in enumerate(info_predictions.items()):
            start_frame_idx = info_sliding_windows[sliding_window]['start_frame_idx']
            end_frame_idx = info_sliding_windows[sliding_window]['end_frame_idx']
            prediction_idx = np.argmax(prediction) 

            if prediction_idx == 0:
                continue

            predictions.append((start_frame_idx, prediction_idx, prediction[prediction_idx], sliding_window))

        gts = [] # tuple
        for frame in info_actions.keys():
            action_name = info_actions[frame]['action_name']
            frame_idx = int(frame.split('_')[-1])

            gts.append((frame_idx, 1 if action_name == 'pass' else 2))
        gts.sort(key=lambda x: x[0])

        return gts, predictions

    def _compute_expected_goals(self, game_name):
        # predictions for expected goals
        prefix = 'sliding_window_videos/sliding_window_videos_expected_goals'
        output_dir = os.path.join(self._root_path, game_name, 'sliding_window_videos/sliding_window_videos_information_expected_goals')

        gts, predictions = self._compute_predictions(game_name)
        predictions_nms = self._apply_nms(gts, predictions)
        predictions_nms = [pred for pred in predictions_nms if pred[1] == 2]

        with open(os.path.join(output_dir, 'test.csv'), 'w') as f:
            for pred in predictions_nms:
                start_frame_idx = pred[0] - 20 # back 20 frames so the video stops right when the player performs the shot

                expected_goals_videos = sorted(os.listdir(os.path.join(self._root_path, game_name, prefix)))
                for expected_goals_video in expected_goals_videos:
                    video_start_idx = int(expected_goals_video.split('_')[-2])
                    if start_frame_idx == video_start_idx:
                        video_name = expected_goals_video
                        break

                f.write('{}/{} -1\n'.format(prefix, video_name))

        # gts for expected goals
        prefix = 'gt_videos'
        output_dir = os.path.join(self._root_path, game_name, 'gt_videos_observations')

        with open(os.path.join(output_dir, 'test.csv'), 'w') as f:
            for video_name in sorted(os.listdir(os.path.join(self._root_path, game_name, 'gt_videos'))):
                if '_'.join(video_name.split('_')[1:-1]) == 'expected_goals':
                    f.write('{}/{} -1\n'.format(prefix, video_name))

    def _plot_expected_goals(self, game_name):
        SMM_WIDTH = 96
        SMM_HEIGHT = 72

        MINIMAP_NORM_X_MIN = -1.0
        MINIMAP_NORM_X_MAX = 1.0
        MINIMAP_NORM_Y_MIN = -1.0 / 2.25
        MINIMAP_NORM_Y_MAX = 1.0 / 2.25

        expected_goals_information = json.load(open(os.path.join(self._root_path, game_name,
                                                                 'sliding_window_videos/sliding_window_videos_information_expected_goals/sliding_window_videos_information.json'), 'r'))
        expected_goals_predictions = json.load(open(os.path.join(self._root_path, game_name,
                                                                 'sliding_window_videos/sliding_window_videos_information_expected_goals/sliding_window_videos_predictions.json'), 'r'))
        football_observations = json.load(open(os.path.join(self._root_path, game_name, 'football_observations.json'), 'r'))

        radar = Image.open('files/radar.bmp').convert('RGB')
        frame = np.array(radar.resize((SMM_WIDTH, SMM_HEIGHT)))

        mask = np.all(frame == [128, 128, 128], axis=-1)
        frame[mask] = np.array([0, 202, 0])

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(frame)
        ax[1].imshow(frame)

        for key, value in expected_goals_predictions.items():
            prob_goal = value[1]

            shot_frame = 'frame_{}'.format(expected_goals_information[key]['end_frame_idx'])
            ball_x = football_observations[shot_frame]['ball']['position']['x'] / 54.4
            ball_y = football_observations[shot_frame]['ball']['position']['y'] / -83.6

            ball_x = (ball_x - MINIMAP_NORM_X_MIN) / (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame.shape[1]
            ball_y = (ball_y - MINIMAP_NORM_Y_MIN) / (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame.shape[0]

            print(ball_x, ball_y, prob_goal, key)
            radius = prob_goal * (2 - 0.5) + 0.5 # radius [10, 50]

            circle = Circle((ball_x, ball_y), radius)
            ax[0].add_patch(circle)

        expected_goals_gts_information = json.load(open(os.path.join(self._root_path, game_name, 'gt_videos_observations/ground_truth_videos_information.json'), 'r'))
        expected_goals_gts_predictions = json.load(open(os.path.join(self._root_path, game_name, 'gt_videos_observations/sliding_window_videos_predictions.json'), 'r'))

        for key, value in expected_goals_gts_predictions.items():
            prob_goal = value[1]

            video_name = key.split('_')[-1]
            shot_frame = 'frame_{}'.format(expected_goals_gts_information['expected_goals'][video_name]['end_frame_idx'])

            ball_x = football_observations[shot_frame]['ball']['position']['x'] / 54.4
            ball_y = football_observations[shot_frame]['ball']['position']['y'] / -83.6

            ball_x = (ball_x - MINIMAP_NORM_X_MIN) / (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame.shape[1]
            ball_y = (ball_y - MINIMAP_NORM_Y_MIN) / (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame.shape[0]

            print(ball_x, ball_y, prob_goal, key)
            radius = prob_goal * (2 - 0.5) + 0.5 # radius [10, 50]

            circle = Circle((ball_x, ball_y), radius)
            ax[1].add_patch(circle)

        ax[0].set_title('Predictions')
        ax[1].set_title('GTs')

        plt.show()

    def _validation(self):
        with open(self._games_txt, 'r') as f:
            for game_name in f:
                game_name = game_name.strip()

                for w_assign_to_gt in [10, 15, 20]:
                    for w_class_value in [10, 15, 20]:
                        for th_class_value in [0.85, 0.90, 0.95]:
                            import ipdb; ipdb.set_trace()

    def _test(self):
        with open(self._games_txt, 'r') as f:
            for game_name in f:
                game_name = game_name.strip()

                gts, predictions = self._compute_predictions(game_name)
                predictions_nms = self._apply_nms(gts, predictions)
                # TODO

    def compute_expected_goals(self):
        with open(self._games_txt, 'r') as f:
            for game_name in f:
                game_name = game_name.strip()

                self._compute_expected_goals(game_name)

    def show_highlight(self):
        with open(self._games_txt, 'r') as f:
            for game_name in f:
                game_name = game_name.strip()

                gts, predictions = self._compute_predictions(game_name)
                predictions_nms = self._apply_nms(gts, predictions)

                self._show_plot(gts, predictions)
                self._show_plot(gts, predictions_nms)

    def plot_expected_goals(self):
        with open(self._games_txt, 'r') as f:
            for game_name in f:
                game_name = game_name.strip()

                self._plot_expected_goals(game_name)

    def compute_metrics(self):
        if self._phase == 'validation':
            self._validation()
        elif self._phase == 'test':
            self._test()
