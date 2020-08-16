import numpy as np
from absl import logging

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

    def get_frame_action(self, step_idx, start_observation, observations, num_steps, action, pass_object, goals_object):
        # left/right_action_type = -1 - no action
        # left/right_action_type = 0 - negative example of action
        # left/right_action_type = 1 - positive example of action
        if action == 'pass':
            left_action_type, left_team_start_frame_idx, left_team_end_frame_idx = \
                pass_object.get_frame_pass('left_team', step_idx, start_observation, observations, num_steps)
            right_action_type, right_team_start_frame_idx, right_team_end_frame_idx = \
                pass_object.get_frame_pass('right_team', step_idx, start_observation, observations, num_steps)
        elif action in ['expected_goals', 'shot']:
            left_action_type, left_team_start_frame_idx, left_team_end_frame_idx = \
                goals_object.get_frame_goal('left_team', step_idx, start_observation, observations, num_steps)
            right_action_type, right_team_start_frame_idx, right_team_end_frame_idx = \
                goals_object.get_frame_goal('right_team', step_idx, start_observation, observations, num_steps)
        else:
            raise Exception('Action {} is unknown!'.format(action))

        if left_action_type == -1 and right_action_type == -1:
            return -1, -1, -1

        # assert not (left_action_type != -1 and right_action_type != -1), '{} type is != -1 for both teams!'.format(action)
        if left_action_type != -1 and right_action_type != -1:
            logging.warn('{} action is != -1 for both teams! Skipping this action!'.format(action))
            return -1, -1, -1

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
