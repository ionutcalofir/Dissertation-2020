import json
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from absl import logging
from PIL import Image

from football_utils.football_constants import *
from football_utils.observations import Observations
from football_utils.frames import Frames
from football_utils.game_engine import GameEngine
from football_utils.game_pass import GamePass
from football_utils.game_goals import GameGoals
from football_utils.video import Video

class DatasetGeneration:
    def __init__(self,
                 dataset_name,
                 dataset_path,
                 dataset_output_path,
                 video_recognition_classes,
                 downscale_videos):
        random.seed(42)

        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        self._dataset_output_path = dataset_output_path
        self._video_recognition_classes = video_recognition_classes
        self._downscale_videos = downscale_videos

        self._observations = Observations()
        self._frames = Frames()
        self._game_engine = GameEngine()
        self._pass = GamePass()
        self._goals = GameGoals()
        self._video = Video(self._downscale_videos)

    def _get_frame_action(self, step_idx, start_observation, observations, num_steps, action):
        # left/right_action_type = -1 - no action
        # left/right_action_type = 0 - negative example of action
        # left/right_action_type = 1 - positive example of action
        if action == 'pass':
            left_action_type, left_team_start_frame_idx, left_team_end_frame_idx = \
                self._pass.get_frame_pass('left_team', step_idx, start_observation, observations, num_steps)
            right_action_type, right_team_start_frame_idx, right_team_end_frame_idx = \
                self._pass.get_frame_pass('right_team', step_idx, start_observation, observations, num_steps)
        elif action in ['expected_goals', 'shot']:
            left_action_type, left_team_start_frame_idx, left_team_end_frame_idx = \
                self._goals.get_frame_goal('left_team', step_idx, start_observation, observations, num_steps)
            right_action_type, right_team_start_frame_idx, right_team_end_frame_idx = \
                self._goals.get_frame_goal('right_team', step_idx, start_observation, observations, num_steps)
        else:
            raise Exception('Action {} is unknown!'.format(action))

        if left_action_type == -1 and right_action_type == -1:
            return -1, -1, -1

        assert not (left_action_type != -1 and right_action_type != -1), '{} type is != -1 for both teams!'.format(action)

        if left_action_type != -1:
            if left_action_type == 1:
                return 1, left_team_start_frame_idx, left_team_end_frame_idx
            else:
                return 0, left_team_start_frame_idx, left_team_end_frame_idx
        else:
            if right_action_type == 1:
                return 1, right_team_start_frame_idx, right_team_end_frame_idx
            else:
                return 0, right_team_start_frame_idx, right_team_end_frame_idx

    def _generate_dataset_action(self, action):
        os.makedirs(self._dataset_output_path, exist_ok=True)

        if self._dataset_name == 'video_recognition':
            action_paths = {cls : os.path.join(self._dataset_output_path, cls) for cls in self._video_recognition_classes}
            action_paths['no_action'] = os.path.join(self._dataset_output_path, 'no_action')
        elif action == 'expected_goals':
            action_paths = {cls : os.path.join(self._dataset_output_path, cls) for cls in ['0', '1']}

        for action_path in action_paths.values():
            os.makedirs(action_path, exist_ok=True)

        for dump_nr, dump_name in enumerate(sorted(os.listdir(self._dataset_path))):
            logging.info('Preprocess dump {} ({}/{})'.format(dump_name, dump_nr + 1, len(os.listdir(self._dataset_path))))
            dump_path = os.path.join(self._dataset_path, dump_name)

            frames_path = self._frames.get_frames_path(dump_path)
            observations = self._observations.get_observations(dump_path)

            if self._dataset_name == 'video_recognition':
                action_frames = {cls : [] for cls in self._video_recognition_classes}
                action_frames['no_action'] = []
            elif action == 'expected_goals':
                action_frames = {cls : [] for cls in ['0', '1']}

            for step_idx in range(len(observations)):
                start_observation = self._observations.get_observation(step_idx, observations)

                if self._dataset_name == 'video_recognition':
                    for action in self._video_recognition_classes:
                        action_type, start_frame_idx, end_frame_idx = \
                            self._get_frame_action(step_idx, start_observation, observations, min(DATASET_GENERATION_FRAMES_WINDOW, len(observations) - 1 - step_idx), action=action)

                        if action_type == -1:
                            continue

                        if action in ['pass', 'shot']:
                            start_frame_idx = max(0, start_frame_idx // STEPS_PER_FRAME - 2) # -2 frame back for more information
                            end_frame_idx = min(end_frame_idx // STEPS_PER_FRAME + 3, len(observations) // STEPS_PER_FRAME - 1) # +3 frame back for more information

                        if ((action_type == 1) or (action_type == 0 and action == 'shot')) \
                                and not (len(action_frames[action]) > 0 and start_frame_idx == action_frames[action][-1][0]):
                            action_frames[action].append((start_frame_idx, end_frame_idx))

                        # For `pass` action hard negative examples will be provided. (When a player
                        # performs a pass, but the ball is intercepted by the other team)
                        if action == 'pass' and action_type == 0 \
                                and not (len(action_frames['no_action']) > 0 and start_frame_idx == action_frames['no_action'][-1][0]):
                            action_frames['no_action'].append((start_frame_idx, end_frame_idx))
                elif action == 'expected_goals':
                    action_type, start_frame_idx, end_frame_idx = \
                        self._get_frame_action(step_idx, start_observation, observations, min(DATASET_GENERATION_FRAMES_WINDOW, len(observations) - 1 - step_idx), action=action)

                    if action_type == -1:
                        continue

                    end_frame_idx = start_frame_idx // STEPS_PER_FRAME # it stops right when the player performs the shot (start_frame_idx is not put there by mistake)
                    start_frame_idx = max(0, start_frame_idx // STEPS_PER_FRAME - 20) # -20 frames back before the player performs the shot

                    if action_type == 0 \
                            and not (len(action_frames['0']) > 0 and start_frame_idx == action_frames['0'][-1][0]):
                        action_frames['0'].append((start_frame_idx, end_frame_idx))
                    elif action_type == 1 \
                            and not (len(action_frames['1']) > 0 and start_frame_idx == action_frames['1'][-1][0]):
                        action_frames['1'].append((start_frame_idx, end_frame_idx))

            def save_videos(i, action_frame, cls, num_examples, random_examples=False):
                print('Preprocess class {} frames {}/{}'.format(cls, i + 1, num_examples))
                video_path = os.path.join(action_paths[cls], '{}_video_{}'.format(dump_name, i + 1))
                if cls == 'no_action' and self._dataset_name == 'video_recognition' and not random_examples:
                    video_path += '_hard_pass'
                frames = [self._frames.get_frame(frames_path[frame]) for frame in range(action_frame[0], action_frame[1])]
                self._video.dump_video(video_path, frames)

            for cls in action_frames:
                Parallel(n_jobs=-2)(delayed(save_videos)(i, action_frame, cls, len(action_frames[cls])) for i, action_frame in enumerate(action_frames[cls]))

            # Add random frames for the `no_action` class
            if self._dataset_name == 'video_recognition':
                all_frames = [action_frame for cls in action_frames for action_frame in action_frames[cls]]
                num_examples = sum([len(action_frames[cls]) for cls in action_frames if cls != 'no_action']) // (len(action_frames) - 1)
                min_diff = min([action_frame[1] - action_frame[0] for action_frame in all_frames])
                max_diff = max([action_frame[1] - action_frame[0] for action_frame in all_frames])

                lim_search = 10 # maximum number of attempts to generate a valid example
                examples_frames = []
                for i, example in enumerate(range(num_examples)):
                    for step in range(lim_search):
                        start_frame = math.floor(random.uniform(0, len(observations) // STEPS_PER_FRAME))
                        r = math.floor(random.uniform(min_diff, max_diff))
                        end_frame = min(start_frame + r, len(observations) // STEPS_PER_FRAME - 1)

                        is_ok = True
                        for frames_range in all_frames:
                            if (frames_range[0] <= start_frame and start_frame <= frames_range[1]) \
                                    or (frames_range[0] <= end_frame and end_frame <= frames_range[1]) \
                                    or (start_frame <= frames_range[0] and frames_range[1] <= end_frame):
                                is_ok = False
                                break
                        for frames_range in examples_frames:
                            if (frames_range[0] <= start_frame and start_frame <= frames_range[1]) \
                                    or (frames_range[0] <= end_frame and end_frame <= frames_range[1]) \
                                    or (start_frame <= frames_range[0] and frames_range[1] <= end_frame):
                                is_ok = False
                                break

                        if is_ok:
                            examples_frames.append((start_frame, end_frame))
                            break
                logging.info('Generated examples {}/{} for `no_action` class!'.format(len(examples_frames), num_examples))

                Parallel(n_jobs=-2)(delayed(save_videos)(i, action_frame, 'no_action', len(examples_frames), random_examples=True) for i, action_frame in enumerate(examples_frames))

            logging.info('Done!\n')

    def _generate_dataset_heatmap(self):
        os.makedirs(self._dataset_output_path, exist_ok=True)
        cmap_name = 'viridis'

        for dump_name in sorted(os.listdir(self._dataset_path)):
            logging.info('Preprocess dump {}'.format(dump_name))
            dump_path = os.path.join(self._dataset_path, dump_name)

            frames_path = self._frames.get_frames_path(dump_path)
            observations = self._observations.get_observations(dump_path)

            frame = self._frames.get_frame(frames_path[0])
            frames = []
            heatmap = np.zeros(frame.shape[:2])
            for frame_nr, frame_path in enumerate(frames_path):
                frames.append(self._frames.get_frame(frame_path))

                if (frame_nr + 1) % 100 == 0:
                    logging.info('Preprocess frame: {}/{}'.format(frame_nr + 1, len(frames_path)))
                now_step = 'step_{}'.format(frame_nr * 10 + 9)
                ball_point = (observations[now_step]['ball']['position_projected']['x'],
                              observations[now_step]['ball']['position_projected']['y'])

                r = 10
                center_y, center_x = ball_point[1], ball_point[0]
                y, x = np.ogrid[-center_y:heatmap.shape[0] - center_y, -center_x:heatmap.shape[1] - center_x]
                mask = x*x + y*y <= r*r
                heatmap[mask] += 1.

                if (frame_nr + 1) % 200 == 0:
                    heatmap_path = os.path.join(self._dataset_output_path, 'heatmap_{}.png'.format((frame_nr + 1) // 200))
                    heatmap_normalized = heatmap / np.max(heatmap)
                    fig, ax = plt.subplots(figsize=(14, 7.5))
                    plt.imshow(frame)
                    plt.imshow(heatmap_normalized, cmap=cmap_name, alpha=0.5)
                    plt.savefig(heatmap_path, bbox_inches='tight')

                    # video_path = os.path.join(self._dataset_output_path, '{}_video_{}'.format(dump_name, (frame_nr + 1) // 200))
                    # self._video.dump_video(video_path, frames)
                    # frames = []

            if len(frames_path) % 200 != 0:
                heatmap_path = os.path.join(self._dataset_output_path, 'heatmap_{}.png'.format(math.ceil(len(frames_path) / 200)))
                heatmap_normalized = heatmap / np.max(heatmap)
                fig, ax = plt.subplots(figsize=(14, 7.5))
                plt.imshow(frame)
                plt.imshow(heatmap_normalized, cmap=cmap_name, alpha=0.5)
                plt.savefig(heatmap_path, bbox_inches='tight')

                # video_path = os.path.join(self._dataset_output_path, '{}_video_{}'.format(dump_name, math.ceil(len(frames_path) / 200)))
                # self._video.dump_video(video_path, frames)
                # frames = []

        logging.info('Done!')

    def generate(self):
        if self._dataset_name == 'video_recognition':
            self._generate_dataset_action(action=None)
        elif self._dataset_name == 'expected_goals':
            self._generate_dataset_action(action='expected_goals')
        elif self._dataset_name == 'heatmap':
            self._generate_dataset_heatmap()
        else:
            logging.error('Dataset generation for {} is not currently implemented'.format(self._dataset_name))
