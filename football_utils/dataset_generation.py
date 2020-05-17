import json
import os
import numpy as np
from absl import logging
from PIL import Image

from football_utils.football_constants import *
from football_utils.observations import Observations
from football_utils.frames import Frames
from football_utils.game_engine import GameEngine
from football_utils.game_pass import GamePass
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
        self._video = Video()

    def _generate_dataset_pass(self):
        os.makedirs(self._dataset_output_path, exist_ok=True)

        passes_path = os.path.join(self._dataset_output_path, '1')
        non_passes_path = os.path.join(self._dataset_output_path, '0')
        os.makedirs(passes_path, exist_ok=True)
        os.makedirs(non_passes_path, exist_ok=True)

        for dump_name in sorted(os.listdir(self._dataset_path)):
            logging.info('Preprocess dump {}'.format(dump_name))
            dump_path = os.path.join(self._dataset_path, dump_name)

            frames_path = self._frames.get_frames_path(dump_path)
            observations = self._observations.get_observations(dump_path)

            nr_observations_wo_agent = len(observations) - len(frames_path)

            pass_frames = []
            non_pass_frames = []

            # Agent action case
            for frame_idx in range(len(frames_path)):
                step_idx = frame_idx * STEPS_PER_FRAME - 1
                start_observation = observations['agent_action_frame_{}'.format(frame_idx)]

                left_pass_type, left_team_start_frame_idx, left_team_end_frame_idx = \
                    self._pass.get_frame_pass('left_team', step_idx, start_observation, observations, min(300, nr_observations_wo_agent - 1 - step_idx))
                right_pass_type, right_team_start_frame_idx, right_team_end_frame_idx = \
                    self._pass.get_frame_pass('right_team', step_idx, start_observation, observations, min(300, nr_observations_wo_agent - 1 - step_idx))

                if left_pass_type == -1 and right_pass_type == -1:
                    continue

                assert not (left_pass_type != -1 and right_pass_type != -1), 'End frame idx is != -1 for both teams!'

                if left_pass_type != -1:
                    if left_pass_type == 1:
                        pass_frames.append((left_team_start_frame_idx, left_team_end_frame_idx))
                    else:
                        non_pass_frames.append((left_team_start_frame_idx, left_team_end_frame_idx))
                else:
                    if right_pass_type == 1:
                        pass_frames.append((right_team_start_frame_idx, right_team_end_frame_idx))
                    else:
                        non_pass_frames.append((right_team_start_frame_idx, right_team_end_frame_idx))

            for step_idx in range(nr_observations_wo_agent):
                start_observation = self._observations.get_observation(step_idx, observations)
                left_pass_type, left_team_start_frame_idx, left_team_end_frame_idx = \
                    self._pass.get_frame_pass('left_team', step_idx, start_observation, observations, min(300, nr_observations_wo_agent - 1 - step_idx))
                right_pass_type, right_team_start_frame_idx, right_team_end_frame_idx = \
                    self._pass.get_frame_pass('right_team', step_idx, start_observation, observations, min(300, nr_observations_wo_agent - 1 - step_idx))

                if left_pass_type == -1 and right_pass_type == -1:
                    continue

                assert not (left_pass_type != -1 and right_pass_type != -1), 'End frame idx is != -1 for both teams!'

                if left_pass_type != -1:
                    if left_pass_type == 1:
                        pass_frames.append((left_team_start_frame_idx, left_team_end_frame_idx))
                    else:
                        non_pass_frames.append((left_team_start_frame_idx, left_team_end_frame_idx))
                else:
                    if right_pass_type == 1:
                        pass_frames.append((right_team_start_frame_idx, right_team_end_frame_idx))
                    else:
                        non_pass_frames.append((right_team_start_frame_idx, right_team_end_frame_idx))

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
        # TODO:
        #   - generate goal videos
        #   - generate no goal videos
        #       - get frames where the player performs a shot
        pass

    def _generate_dataset_heatmap(self):
        # TODO:
        #   - generate heatmaps at some fixed intervals from a match
        pass

    def generate(self):
        if self._dataset_name == 'pass':
            self._generate_dataset_pass()
        elif self._dataset_name == 'expected_goals':
            self._generate_dataset_expected_goals()
        elif self._dataset_name == 'heatmap':
            self._generate_dataset_heatmap()
        else:
            logging.error('Dataset generation for {} is not currently implemented'.format(self._dataset_name))
