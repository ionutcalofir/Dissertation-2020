import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
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
                 downscale_videos):
        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        self._dataset_output_path = dataset_output_path
        self._downscale_videos = downscale_videos

        self._observations = Observations()
        self._frames = Frames()
        self._game_engine = GameEngine()
        self._pass = GamePass()
        self._goals = GameGoals()
        self._video = Video()

    def _get_frame_action(self, step_idx, start_observation, observations, num_steps, action):
        # left/right_action_type = -1 - no action
        # left/right_action_type = 0 - negative example of action
        # left/right_action_type = 1 - positive example of action
        if action == 'pass':
            left_action_type, left_team_start_frame_idx, left_team_end_frame_idx = \
                self._pass.get_frame_pass('left_team', step_idx, start_observation, observations, num_steps)
            right_action_type, right_team_start_frame_idx, right_team_end_frame_idx = \
                self._pass.get_frame_pass('right_team', step_idx, start_observation, observations, num_steps)
        elif action in ['shot_goal', 'shot']:
            left_action_type, left_team_start_frame_idx, left_team_end_frame_idx = \
                self._goals.get_frame_goal('left_team', step_idx, start_observation, observations, num_steps)
            right_action_type, right_team_start_frame_idx, right_team_end_frame_idx = \
                self._goals.get_frame_goal('right_team', step_idx, start_observation, observations, num_steps)

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

        positive_action_path = os.path.join(self._dataset_output_path, '1')
        negative_action_path = os.path.join(self._dataset_output_path, '0')
        os.makedirs(positive_action_path, exist_ok=True)
        os.makedirs(negative_action_path, exist_ok=True)

        for dump_name in sorted(os.listdir(self._dataset_path)):
            logging.info('Preprocess dump {}'.format(dump_name))
            dump_path = os.path.join(self._dataset_path, dump_name)

            frames_path = self._frames.get_frames_path(dump_path)
            observations = self._observations.get_observations(dump_path)

            positive_action_frames = []
            negative_action_frames = []

            for step_idx in range(len(observations)):
                start_observation = self._observations.get_observation(step_idx, observations)

                action_type, start_frame_idx, end_frame_idx = \
                        self._get_frame_action(step_idx, start_observation, observations, min(200, len(observations) - 1 - step_idx), action=action)

                if action_type == -1:
                    continue

                if action in ['pass', 'shot']:
                    start_frame_idx = start_frame_idx // STEPS_PER_FRAME - 1 # -1 frame back for more information
                    end_frame_idx = end_frame_idx // STEPS_PER_FRAME + 1 # +1 frame back for more information
                elif action == 'shot_goal':
                    end_frame_idx = start_frame_idx // STEPS_PER_FRAME - 2 # -2 frame back so it stops right when the player performs the shot (start_frame_idx is not put there by mistake)
                    start_frame_idx = start_frame_idx // STEPS_PER_FRAME - 20 # -20 frames back before the player performs the shot

                if (len(positive_action_frames) > 0 and start_frame_idx == positive_action_frames[-1][0]) \
                        or (len(negative_action_frames) > 0 and start_frame_idx == negative_action_frames[-1][0]):
                    continue

                if action_type == 0:
                    negative_action_frames.append((start_frame_idx, end_frame_idx))
                else:
                    positive_action_frames.append((start_frame_idx, end_frame_idx))

            for i, positive_action_frame in enumerate(positive_action_frames):
                logging.info('Preprocess {} frames {}/{}'.format(action, i + 1, len(positive_action_frames)))
                video_path = os.path.join(positive_action_path, '{}_video_{}'.format(dump_name, i + 1))
                frames = [self._frames.get_frame(frames_path[frame]) for frame in range(positive_action_frame[0] - 1, positive_action_frame[1] + 2)]
                self._video.dump_video(video_path, frames)

            print()
            for i, negative_action_frame in enumerate(negative_action_frames):
                logging.info('Preprocess no {} frames {}/{}'.format(action, i + 1, len(negative_action_frames)))
                video_path = os.path.join(negative_action_path, '{}_video_{}'.format(dump_name, i + 1))
                frames = [self._frames.get_frame(frames_path[frame]) for frame in range(negative_action_frame[0] - 1, negative_action_frame[1] + 2)]
                self._video.dump_video(video_path, frames)

            logging.info('Done!')

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
        if self._dataset_name == 'pass':
            self._generate_dataset_action(action='pass')
        elif self._dataset_name == 'shot':
            self._generate_dataset_action(action='shot')
        elif self._dataset_name == 'shot_goal':
            self._generate_dataset_action(action='shot_goal')
        elif self._dataset_name == 'heatmap':
            self._generate_dataset_heatmap()
        else:
            logging.error('Dataset generation for {} is not currently implemented'.format(self._dataset_name))
