import os
import csv
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
        self._w_class = {1: 10,
                         2: 20}
        self._th_class = {1: 0.85,
                          2: 0.85}

        self._w_assign_to_gt = 20

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
        ax.set_xlabel('Frames')
        ax.set_ylabel('Probability of event')
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

    def _assign_action_to_gts(self, gts, predictions_nms, w_assign_to_gt):
        assigned = []
        predictions_used_frames = {}
        for gt in gts:
            lt = gt[0] - w_assign_to_gt
            rt = gt[0] + w_assign_to_gt

            min_dist = 2 * w_assign_to_gt + 1
            idx = -1
            for i, prediction_nms in enumerate(predictions_nms):
                if lt <= prediction_nms[0] and prediction_nms[0] <= rt \
                        and abs(gt[0] - prediction_nms[0]) < min_dist \
                        and prediction_nms[0] not in predictions_used_frames:
                    min_dist = abs(gt[0] - prediction_nms[0])
                    idx = i

            if idx != -1:
                predictions_used_frames[predictions_nms[idx][0]] = True
                assigned.append((gt[0], predictions_nms[idx][0]))

        tp = len(assigned)
        fp = max(0, len(predictions_nms) - len(assigned))
        fn = len(gts) - len(assigned)

        try:
            precision = tp / (tp + fp)
        except:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except:
            recall = 0
        try:
            f1 = 2 * ((precision * recall) / (precision + recall))
        except:
            f1 = 0

        return f1, precision, recall, tp, fp, fn, assigned

    def _assign_to_gts(self, gts, predictions_nms, w_assign_to_gt):
        pass_gts = [gt for gt in gts if gt[1] == 1]
        shot_gts = [gt for gt in gts if gt[1] == 2]

        pass_predictions_nms = [pred for pred in predictions_nms if pred[1] == 1]
        shot_predictions_nms = [pred for pred in predictions_nms if pred[1] == 2]

        pass_f1, pass_precision, pass_recall, pass_tp, pass_fp, pass_fn, pass_assigned = self._assign_action_to_gts(pass_gts, pass_predictions_nms, w_assign_to_gt)
        shot_f1, shot_precision, shot_recall, shot_tp, shot_fp, shot_fn, shot_assigned = self._assign_action_to_gts(shot_gts, shot_predictions_nms, w_assign_to_gt)

        print('Num passes {}'.format(len(pass_gts)))
        print('Num shots {}'.format(len(shot_gts)))

        return (pass_f1, pass_precision, pass_recall, pass_tp, pass_fp, pass_fn), \
               (shot_f1, shot_precision, shot_recall, shot_tp, shot_fp, shot_fn), \
               (pass_assigned, shot_assigned)

    def _compute_avg_metrics(self, precision, recall, f1, no_games):
        avg_precision = sum(precision) / no_games
        avg_recall = sum(recall) / no_games
        avg_f1 = sum(f1) / no_games

        return avg_precision, avg_recall, avg_f1

    def _validation(self):
        window_lengths = [10, 15, 20]

        w_class_values = []
        th_class_values = []
        for i in [10, 15, 20]:
            for j in [10, 15, 20]:
                w_class_values.append((i, j))
        for i in [0.85, 0.90, 0.95]:
            for j in [0.85, 0.90, 0.95]:
                th_class_values.append((i, j))

        w_assign_to_gt_values = [10, 15, 20]

        iterations = len(window_lengths) \
                    * len(w_class_values) \
                    * len(th_class_values) \
                    * len(w_assign_to_gt_values)
        idx = 0

        validation_dict = {}
        for window_length in window_lengths:
            for w_assign_to_gt in w_assign_to_gt_values:
                for w_class_value in w_class_values:
                    for th_class_value in th_class_values:
                        idx += 1
                        print('Preprocess iteration {}/{}'.format(idx, iterations))

                        no_games = 0
                        precisions = []
                        recalls = []
                        f1s = []
                        num_passes = 0
                        num_shots = 0
                        with open(self._games_txt, 'r') as f:
                            for game_name in f:
                                no_games += 1
                                game_name = game_name.strip()
                                print('Preprocess game: {}'.format(game_name))

                                gts, predictions = self._compute_predictions(game_name, window_length)
                                num_passes += len([gt for gt in gts if gt[1] == 1])
                                num_shots += len([gt for gt in gts if gt[1] == 2])


                                w_class = {1: w_class_value[0],
                                           2: w_class_value[1]}
                                th_class = {1: th_class_value[0],
                                            2: th_class_value[1]}
                                predictions_nms = self._apply_nms(gts, predictions, w_class, th_class)

                                (pass_f1, pass_precision, pass_recall, _, _, _), \
                                (shot_f1, shot_precision, shot_recall, _, _, _), \
                                (_, _) = \
                                        self._assign_to_gts(gts, predictions_nms, w_assign_to_gt)

                                precisions.append((pass_precision, shot_precision))
                                recalls.append((pass_recall, shot_recall))
                                f1s.append((pass_f1, shot_f1))

                        key = 'window_length_{}_w_assign_to_gt_{}_w_class_value_{}_th_class_value_{}'.format(window_length,
                                                                                                             w_assign_to_gt,
                                                                                                             w_class_value,
                                                                                                             th_class_value)

                        pass_avg_precision, pass_avg_recall, pass_avg_f1 = self._compute_avg_metrics([precision[0] for precision in precisions],
                                                                                                     [recall[0] for recall in recalls],
                                                                                                     [f1[0] for f1 in f1s],
                                                                                                     no_games)
                        shot_avg_precision, shot_avg_recall, shot_avg_f1 = self._compute_avg_metrics([precision[1] for precision in precisions],
                                                                                                     [recall[1] for recall in recalls],
                                                                                                     [f1[1] for f1 in f1s],
                                                                                                     no_games)

                        validation_dict[key] = {}
                        validation_dict[key]['pass'] = {}
                        validation_dict[key]['pass']['precision'] = pass_avg_precision
                        validation_dict[key]['pass']['recall'] = pass_avg_recall
                        validation_dict[key]['pass']['f1'] = pass_avg_f1
                        validation_dict[key]['shot'] = {}
                        validation_dict[key]['shot']['precision'] = shot_avg_precision
                        validation_dict[key]['shot']['recall'] = shot_avg_recall
                        validation_dict[key]['shot']['f1'] = shot_avg_f1
                        validation_dict[key]['f1_weighted'] = (num_passes * pass_avg_f1 + num_shots * shot_avg_f1) / (num_passes + num_shots)

        with open(os.path.join(self._root_path, 'configs', 'validation.csv'), 'w') as f:
            fieldnames = ['VALIDATION PARAMS',
                          'PASS AVG PRECISION', 'PASS AVG RECALL', 'PASS AVG F1',
                          'SHOT AVG PRECISION', 'SHOT AVG RECALL', 'SHOT AVG F1',
                          'F1 WEIGHTED']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for key in validation_dict.keys():
                writer.writerow({'VALIDATION PARAMS': key,
                                 'PASS AVG PRECISION': validation_dict[key]['pass']['precision'],
                                 'PASS AVG RECALL': validation_dict[key]['pass']['recall'],
                                 'PASS AVG F1': validation_dict[key]['pass']['f1'],
                                 'SHOT AVG PRECISION': validation_dict[key]['shot']['precision'],
                                 'SHOT AVG RECALL': validation_dict[key]['shot']['recall'],
                                 'SHOT AVG F1': validation_dict[key]['shot']['f1'],
                                 'F1 WEIGHTED': validation_dict[key]['f1_weighted']})

    def _test(self):
        test_dict = {}
        no_games = 0
        precisions = []
        recalls = []
        f1s = []
        num_passes = 0
        num_shots = 0
        with open(self._games_txt, 'r') as f:
            for game_name in f:
                no_games += 1
                game_name = game_name.strip()
                print('Preprocess game: {}'.format(game_name))

                gts, predictions = self._compute_predictions(game_name)
                predictions_nms = self._apply_nms(gts, predictions)
                num_passes += len([gt for gt in gts if gt[1] == 1])
                num_shots += len([gt for gt in gts if gt[1] == 2])

                (pass_f1, pass_precision, pass_recall, _, _, _), \
                (shot_f1, shot_precision, shot_recall, _, _, _), \
                (_, _) = \
                        self._assign_to_gts(gts, predictions_nms, self._w_assign_to_gt)

                precisions.append((pass_precision, shot_precision))
                recalls.append((pass_recall, shot_recall))
                f1s.append((pass_f1, shot_f1))

        key = 'window_length_{}_w_assign_to_gt_{}_w_class_value_{}_th_class_value_{}'.format(self._window_length,
                                                                                             self._w_assign_to_gt,
                                                                                             (self._w_class[1], self._w_class[2]),
                                                                                             (self._th_class[1], self._th_class[2]))

        pass_avg_precision, pass_avg_recall, pass_avg_f1 = self._compute_avg_metrics([precision[0] for precision in precisions],
                                                                                     [recall[0] for recall in recalls],
                                                                                     [f1[0] for f1 in f1s],
                                                                                     no_games)
        shot_avg_precision, shot_avg_recall, shot_avg_f1 = self._compute_avg_metrics([precision[1] for precision in precisions],
                                                                                     [recall[1] for recall in recalls],
                                                                                     [f1[1] for f1 in f1s],
                                                                                     no_games)

        test_dict[key] = {}
        test_dict[key]['pass'] = {}
        test_dict[key]['pass']['precision'] = pass_avg_precision
        test_dict[key]['pass']['recall'] = pass_avg_recall
        test_dict[key]['pass']['f1'] = pass_avg_f1
        test_dict[key]['shot'] = {}
        test_dict[key]['shot']['precision'] = shot_avg_precision
        test_dict[key]['shot']['recall'] = shot_avg_recall
        test_dict[key]['shot']['f1'] = shot_avg_f1
        test_dict[key]['f1_weighted'] = (num_passes * pass_avg_f1 + num_shots * shot_avg_f1) / (num_passes + num_shots)

        with open(os.path.join(self._root_path, 'configs', 'test.csv'), 'w') as f:
            fieldnames = ['TEST PARAMS',
                          'PASS AVG PRECISION', 'PASS AVG RECALL', 'PASS AVG F1',
                          'SHOT AVG PRECISION', 'SHOT AVG RECALL', 'SHOT AVG F1',
                          'F1 WEIGHTED']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for key in test_dict.keys():
                writer.writerow({'TEST PARAMS': key,
                                 'PASS AVG PRECISION': test_dict[key]['pass']['precision'],
                                 'PASS AVG RECALL': test_dict[key]['pass']['recall'],
                                 'PASS AVG F1': test_dict[key]['pass']['f1'],
                                 'SHOT AVG PRECISION': test_dict[key]['shot']['precision'],
                                 'SHOT AVG RECALL': test_dict[key]['shot']['recall'],
                                 'SHOT AVG F1': test_dict[key]['shot']['f1'],
                                 'F1 WEIGHTED': test_dict[key]['f1_weighted']})

    def compute_expected_goals(self):
        with open(self._games_txt, 'r') as f:
            for game_name in f:
                game_name = game_name.strip()

                self._compute_expected_goals(game_name)

    def show_highlight(self):
        shot_diffs = 0
        pass_diffs = 0
        no_games = 0
        with open(self._games_txt, 'r') as f:
            for game_name in f:
                no_games += 1
                game_name = game_name.strip()

                gts, predictions = self._compute_predictions(game_name)
                predictions_nms = self._apply_nms(gts, predictions)

                (pass_f1, pass_precision, pass_recall, _, _, _), \
                (shot_f1, shot_precision, shot_recall, _, _, _), \
                (pass_assigned, shot_assigned) = \
                        self._assign_to_gts(gts, predictions_nms, self._w_assign_to_gt)

                print(game_name)
                print('Pass f1 {}, precision {}, recall {}'.format(pass_f1, pass_precision, pass_recall))
                print('Shot f1 {}, precision {}, recall {}'.format(shot_f1, shot_precision, shot_recall))

                shot_diffs += sum([abs(sa[1] - sa[0]) for sa in shot_assigned])
                pass_diffs += sum([abs(pa[1] - pa[0]) for pa in pass_assigned])

                # self._show_plot(gts, predictions)
                self._show_plot(gts, predictions_nms)

        print('Avg shot diff per game {}'.format(shot_diffs / no_games))
        print('Avg pass diff per game {}'.format(pass_diffs / no_games))

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
