import json
import os

class Observations:
    def get_observations(self, dump_path):
        observations = json.load(open(os.path.join(dump_path, 'observations.json'), 'r'))
        return observations

    def get_observation(self, step_idx, observations):
        step = 'step_{}'.format(step_idx)
        return observations[step]

    def get_frame_name(self, step_idx, observations):
        return int(observations['step_{}'.format(step_idx)]['frame_name'].split('_')[-1])
