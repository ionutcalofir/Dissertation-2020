import json
import os
import numpy as np
from absl import logging
from PIL import Image

class DatasetGeneration:
    def __init__(self,
                 dataset_name,
                 dataset_path,
                 dataset_output_path):
        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        self._dataset_output_path = dataset_output_path

    def _get_observations(self, dump_path):
        frames_path = os.path.join(dump_path, 'frames')
        frames_path = [os.path.join(frames_path, frame)
                       for frame in sorted(os.listdir(frames_path), key=lambda x: int(x.split('_')[-1][:-4]))]

        observations = json.load(open(os.path.join(dump_path, 'observations.json'), 'r'))

        return frames_path, observations

    def _get_frame(self, frame_path):
        frame = np.array(Image.open(frame_path))

        return frame

    def _get_ball_distance(self, start_position, end_position):
        distance = np.sqrt(np.square(end_position[0] - start_position[0]) + np.square(end_position[1] - start_position[1]))

        return distance

    def _get_last_frame_pass(self, start_frame, observations, num_frames):
        now_step = 'step_{}'.format(start_frame)

        return -1

    def _generate_dataset_pass(self):
        # TODO:
        #   - generate pass videos
        #   - generate non-pass videos
        #       - get random frames that are not in the pass videos
        os.makedirs(self._dataset_output_path, exist_ok=True)

        passes_path = os.path.join(self._dataset_output_path, 'passes')
        non_passes_path = os.path.join(self._dataset_output_path, 'non_passes')
        os.makedirs(passes_path, exist_ok=True)
        os.makedirs(non_passes_path, exist_ok=True)

        for dump_name in sorted(os.listdir(self._dataset_path)):
            dump_path = os.path.join(self._dataset_path, dump_name)

            frames_path, observations = self._get_observations(dump_path)

            for frame_idx in range(0, len(observations)):
                start_frame_idx = int(observations['step_{}'.format(frame_idx)]['frame_name'].split('_')[-1])
                last_frame_idx = self._get_last_frame_pass(frame_idx, observations, min(300, len(observations) - 1 - frame_idx))

                if last_frame_idx == -1:
                    continue

                break

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
