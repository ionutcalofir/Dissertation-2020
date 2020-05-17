import numpy as np

from football_utils.football_constants import *

class GameEngine:
    def get_distance(self, p1, p2, axes='xy'):
        """
        Returns the distance between two points.
        """
        s = 0.
        for axis in axes:
            s += (p2[axis] - p1[axis]) * (p2[axis] - p1[axis])
        return np.sqrt(s)

    def get_magnitude(self, p, axes='xy'):
        """
        Returns the magnitude of a vector.
        """
        p_zero = {axis: 0. for axis in axes}
        return self.get_distance(p_zero, p)

    def is_in_play(self, observation):
        return observation['is_in_play']

    def check_pressed_action(self, team, observation, action):
        if action == 'pass':
            return observation[team]['pressed_action'] in PASS_ACTIONS
        elif action == 'shot':
            return observation[team]['pressed_action'] in SHOT_ACTIONS
        else:
            raise Exception('Action unknown!')
