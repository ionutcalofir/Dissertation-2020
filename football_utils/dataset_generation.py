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
        frames_name = [os.path.join(frames_path, frame)
                       for frame in sorted(os.listdir(frames_path), key=lambda x: int(x.split('_')[-1][:-4]))]

        observations = json.load(open(os.path.join(dump_path, 'observations.json'), 'r'))

        return frames_name, observations

    def _generate_dataset_pass(self):
        os.makedirs(self._dataset_output_path, exist_ok=True)

        passes_path = os.path.join(self._dataset_output_path, 'passes')
        non_passes_path = os.path.join(self._dataset_output_path, 'non_passes')
        os.makedirs(passes_path, exist_ok=True)
        os.makedirs(non_passes_path, exist_ok=True)

        for dump_name in sorted(os.listdir(self._dataset_path)):
            dump_path = os.path.join(self._dataset_path, dump_name)

            frames_name, observations = self._get_observations(dump_path)

            for i, (key, value) in enumerate(observations.items()):
                if value['game_mode'] != 'e_GameMode_Normal':
                    print(key, value['game_mode'])

            logging.info('Done')

    def generate(self):
        if self._dataset_name == 'pass':
            self._generate_dataset_pass()
        else:
            logging.error('Dataset generation for {} is not currently implemented'.format(self._dataset_name))
