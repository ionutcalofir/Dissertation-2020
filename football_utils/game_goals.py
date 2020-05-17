from football_utils.football_constants import *
from football_utils.game_engine import GameEngine

class GameGoals:
    def __init__(self):
        self._game_engine = GameEngine()

    def get_frame_goal(self, team, start_step_idx, start_observation, observations, num_steps):
        team_id = 0 if team == 'left_team' else 1

        # Check if the game is in play mode and the controlled player tries to perform an action
        if not self._game_engine.is_in_play(start_observation)\
                or not self._game_engine.check_pressed_action(team, start_observation, action='shot'):
            return -1, -1, -1

        start_controlled_player = start_observation[team]['controlled_player']
        start_pressed_action = start_observation[team]['pressed_action']

        # Check if the pressed action from the start_observation is the last pressed action for the controlled player
        for step in range(start_step_idx + 1, start_step_idx + num_steps):
            now_step = 'step_{}'.format(step)

            now_controlled_player = observations[now_step][team]['controlled_player']
            now_pressed_action = observations[now_step][team]['pressed_action']

            if now_controlled_player != start_controlled_player:
                break

            if now_pressed_action != 'no_action':
                return -1, -1, -1

        # Get the step where the controlled player last touches the ball
        last_touch_step = -1
        for step in range(max(0, start_step_idx - num_steps), start_step_idx + 100):
            now_step = 'step_{}'.format(step)

            player_touch_ball = observations[now_step]['player_touch_ball']
            team_touch_ball = observations[now_step]['team_touch_ball']

            if team_touch_ball == team_id and player_touch_ball == start_controlled_player:
                last_touch_step = step

        if last_touch_step == -1:
            return -1, -1, -1

        # bb = [(now_step, observations['step_{}'.format(now_step)][team]['pressed_action']) for now_step in range(1000, 1120)]
        # dd = [(now_step, observations['step_{}'.format(now_step)]['player_touch_ball']) for now_step in range(1000, 1120)
        #         if observations['step_{}'.format(now_step)]['player_touch_ball'] != -1]
        # import pdb; pdb.set_trace()

        for step in range(last_touch_step + 1, start_step_idx + num_steps):
            now_step = 'step_{}'.format(step)

            player_touch_ball = observations[now_step]['player_touch_ball']
            team_touch_ball = observations[now_step]['team_touch_ball']
            is_in_play = observations[now_step]['is_in_play']
            is_goal_scored = observations[now_step]['is_goal_scored']

            if team_touch_ball == team_id and player_touch_ball == start_controlled_player:
                return -1, -1, -1

            if (team_touch_ball != -1 and player_touch_ball != -1) \
                    or (not is_in_play):
                if is_in_play:
                    return 0, last_touch_step // STEPS_PER_FRAME - 20, last_touch_step // STEPS_PER_FRAME - 1
                else:
                    if is_goal_scored:
                        return 1, last_touch_step // STEPS_PER_FRAME - 20, last_touch_step // STEPS_PER_FRAME - 1
                    else:
                        return 0, last_touch_step // STEPS_PER_FRAME - 20, last_touch_step // STEPS_PER_FRAME - 1

        return -1, -1, -1
