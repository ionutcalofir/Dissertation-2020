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
                 dataset_output_path):
        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        self._dataset_output_path = dataset_output_path

        self._observations = Observations()
        self._frames = Frames()
        self._game_engine = GameEngine()
        self._pass = GamePass()
        self._goals = GameGoals()
        self._video = Video()

    def _get_frame_pass(self, step_idx, start_observation, observations, num_steps):
        left_pass_type, left_team_start_frame_idx, left_team_end_frame_idx = \
            self._pass.get_frame_pass('left_team', step_idx, start_observation, observations, num_steps)
        right_pass_type, right_team_start_frame_idx, right_team_end_frame_idx = \
            self._pass.get_frame_pass('right_team', step_idx, start_observation, observations, num_steps)

        if left_pass_type == -1 and right_pass_type == -1:
            return -1, -1, -1

        assert not (left_pass_type != -1 and right_pass_type != -1), 'Pass type is != -1 for both teams!'

        if left_pass_type != -1:
            if left_pass_type == 1:
                return 1, left_team_start_frame_idx, left_team_end_frame_idx
            else:
                return 0, left_team_start_frame_idx, left_team_end_frame_idx
        else:
            if right_pass_type == 1:
                return 1, right_team_start_frame_idx, right_team_end_frame_idx
            else:
                return 0, right_team_start_frame_idx, right_team_end_frame_idx

    def _get_frame_goal(self, step_idx, start_observation, observations, num_steps):
        left_shot_goal_type, left_team_start_frame_idx, left_team_end_frame_idx = \
            self._goals.get_frame_goal('left_team', step_idx, start_observation, observations, num_steps)
        right_shot_goal_type, right_team_start_frame_idx, right_team_end_frame_idx = \
            self._goals.get_frame_goal('right_team', step_idx, start_observation, observations, num_steps)

        if left_shot_goal_type == -1 and right_shot_goal_type == -1:
            return -1, -1, -1

        assert not (left_shot_goal_type != -1 and right_shot_goal_type != -1), 'Goal type is != -1 for both teams!'

        if left_shot_goal_type != -1:
            if left_shot_goal_type == 1:
                return 1, left_team_start_frame_idx, left_team_end_frame_idx
            else:
                return 0, left_team_start_frame_idx, left_team_end_frame_idx
        else:
            if right_shot_goal_type == 1:
                return 1, right_team_start_frame_idx, right_team_end_frame_idx
            else:
                return 0, right_team_start_frame_idx, right_team_end_frame_idx

    def _generate_dataset_pass(self):
        os.makedirs(self._dataset_output_path, exist_ok=True)

        passes_path = os.path.join(self._dataset_output_path, '1')
        non_passes_path = os.path.join(self._dataset_output_path, '0')
        os.makedirs(passes_path, exist_ok=True)
        os.makedirs(non_passes_path, exist_ok=True)

        for dump_name in sorted(os.listdir(self._dataset_path)):
            if dump_name != 'dump_20200517-020504163769':
                continue
            logging.info('Preprocess dump {}'.format(dump_name))
            dump_path = os.path.join(self._dataset_path, dump_name)

            frames_path = self._frames.get_frames_path(dump_path)
            observations = self._observations.get_observations(dump_path)

            nr_observations_wo_agent = len(observations) - len(frames_path)

            pass_frames = []
            non_pass_frames = []

            # TODO Agent action case
            # for frame_idx in range(len(frames_path)):
            #     step_idx = frame_idx * STEPS_PER_FRAME - 1
            #     start_observation = observations['agent_action_frame_{}'.format(frame_idx)]

            #     pass_type, start_frame_idx, end_frame_idx = \
            #             self._get_frame_pass(step_idx, start_observation, observations, min(300, nr_observations_wo_agent - 1 - step_idx))

            #     if pass_type == -1:
            #         continue
            #     elif pass_type == 0:
            #         non_pass_frames.append((start_frame_idx, end_frame_idx))
            #     else:
            #         pass_frames.append((start_frame_idx, end_frame_idx))

            for step_idx in range(nr_observations_wo_agent):
                start_observation = self._observations.get_observation(step_idx, observations)

                pass_type, start_frame_idx, end_frame_idx = \
                        self._get_frame_pass(step_idx, start_observation, observations, min(300, nr_observations_wo_agent - 1 - step_idx))

                if pass_type == -1:
                    continue
                elif pass_type == 0:
                    non_pass_frames.append((start_frame_idx, end_frame_idx))
                else:
                    pass_frames.append((start_frame_idx, end_frame_idx))

            for i, pass_frame in enumerate(pass_frames):
                logging.info('Preprocess pass frames {}/{}'.format(i + 1, len(pass_frames)))
                video_path = os.path.join(passes_path, '{}_video_{}'.format(dump_name, i + 1))
                frames = [self._frames.get_frame(frames_path[frame]) for frame in range(pass_frame[0] - 1, pass_frame[1] + 2)]
                self._video.dump_video(video_path, frames)

            print()
            for i, non_pass_frame in enumerate(non_pass_frames):
                logging.info('Preprocess non pass frames {}/{}'.format(i + 1, len(non_pass_frames)))
                video_path = os.path.join(non_passes_path, '{}_video_{}'.format(dump_name, i + 1))
                frames = [self._frames.get_frame(frames_path[frame]) for frame in range(non_pass_frame[0] - 1, non_pass_frame[1] + 2)]
                self._video.dump_video(video_path, frames)

            logging.info('Done!')

    def _generate_dataset_expected_goals(self):
        os.makedirs(self._dataset_output_path, exist_ok=True)

        shot_goal_path = os.path.join(self._dataset_output_path, '1')
        shot_no_goal_path = os.path.join(self._dataset_output_path, '0')
        os.makedirs(shot_goal_path, exist_ok=True)
        os.makedirs(shot_no_goal_path, exist_ok=True)

        for dump_name in sorted(os.listdir(self._dataset_path)):
            if dump_name != 'dump_20200517-205515687273':
                continue
            logging.info('Preprocess dump {}'.format(dump_name))
            dump_path = os.path.join(self._dataset_path, dump_name)

            frames_path = self._frames.get_frames_path(dump_path)
            observations = self._observations.get_observations(dump_path)

            nr_observations_wo_agent = len(observations) - len(frames_path)

            shot_goal_frames = []
            shot_no_goal_frames = []

            # TODO Agent action case
            # for frame_idx in range(len(frames_path)):
            #     step_idx = frame_idx * STEPS_PER_FRAME - 1
            #     start_observation = observations['agent_action_frame_{}'.format(frame_idx)]

            #     shot_goal_type, start_frame_idx, end_frame_idx = \
            #             self._get_frame_goal(step_idx, start_observation, observations, min(300, nr_observations_wo_agent - 1 - step_idx))

            #     if shot_goal_type == -1:
            #         continue
            #     elif shot_goal_type == 0:
            #         shot_no_goal_frames.append((start_frame_idx, end_frame_idx))
            #     else:
            #         shot_goal_frames.append((start_frame_idx, end_frame_idx))

            for step_idx in range(nr_observations_wo_agent):
                start_observation = self._observations.get_observation(step_idx, observations)

                shot_goal_type, start_frame_idx, end_frame_idx = \
                        self._get_frame_goal(step_idx, start_observation, observations, min(300, nr_observations_wo_agent - 1 - step_idx))

                if shot_goal_type == -1:
                    continue
                elif shot_goal_type == 0:
                    shot_no_goal_frames.append((start_frame_idx, end_frame_idx))
                else:
                    shot_goal_frames.append((start_frame_idx, end_frame_idx))

            for i, shot_goal_frame in enumerate(shot_goal_frames):
                logging.info('Preprocess shot goal frames {}/{}'.format(i + 1, len(shot_goal_frames)))
                video_path = os.path.join(shot_goal_path, '{}_video_{}'.format(dump_name, i + 1))
                frames = [self._frames.get_frame(frames_path[frame]) for frame in range(shot_goal_frame[0], shot_goal_frame[1] + 1)]
                self._video.dump_video(video_path, frames)

            print()
            for i, shot_no_goal_frame in enumerate(shot_no_goal_frames):
                logging.info('Preprocess shot no goal frames {}/{}'.format(i + 1, len(shot_no_goal_frames)))
                video_path = os.path.join(shot_no_goal_path, '{}_video_{}'.format(dump_name, i + 1))
                frames = [self._frames.get_frame(frames_path[frame]) for frame in range(shot_no_goal_frame[0], shot_no_goal_frame[1] + 1)]
                self._video.dump_video(video_path, frames)
            logging.info('Done!')

    def _generate_dataset_heatmap(self):
        os.makedirs(self._dataset_output_path, exist_ok=True)

        for dump_name in sorted(os.listdir(self._dataset_path)):
            if dump_name != 'dump_20200517-231054000739':
                continue
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
                    plt.imshow(heatmap_normalized, cmap='hot', alpha=0.5)
                    plt.savefig(heatmap_path, bbox_inches='tight')

                    # video_path = os.path.join(self._dataset_output_path, '{}_video_{}'.format(dump_name, (frame_nr + 1) // 200))
                    # self._video.dump_video(video_path, frames)
                    # frames = []

            if len(frames_path) % 200 != 0:
                heatmap_path = os.path.join(self._dataset_output_path, 'heatmap_{}.png'.format(math.ceil(len(frames_path) / 200)))
                heatmap_normalized = heatmap / np.max(heatmap)
                fig, ax = plt.subplots(figsize=(14, 7.5))
                plt.imshow(frame)
                plt.imshow(heatmap_normalized, cmap='hot', alpha=0.5)
                plt.savefig(heatmap_path, bbox_inches='tight')

                # video_path = os.path.join(self._dataset_output_path, '{}_video_{}'.format(dump_name, math.ceil(len(frames_path) / 200)))
                # self._video.dump_video(video_path, frames)
                # frames = []

        logging.info('Done!')

    def generate(self):
        if self._dataset_name == 'pass':
            self._generate_dataset_pass()
        elif self._dataset_name == 'expected_goals':
            self._generate_dataset_expected_goals()
        elif self._dataset_name == 'heatmap':
            self._generate_dataset_heatmap()
        else:
            logging.error('Dataset generation for {} is not currently implemented'.format(self._dataset_name))
