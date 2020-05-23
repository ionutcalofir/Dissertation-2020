from football_utils.football_constants import *
from football_utils.game_engine import GameEngine

class GamePass:
    def __init__(self):
        self._game_engine = GameEngine()

    def get_frame_pass(self, team, start_step_idx, start_observation, observations, num_steps):
        team_id = 0 if team == 'left_team' else 1

        # Check if the game is in play mode and the controlled player tries to perform an action
        if not self._game_engine.is_in_play(start_observation)\
                or not self._game_engine.check_pressed_action(team, start_observation, action='pass'):
            return -1, -1, -1

        start_controlled_player = start_observation[team]['controlled_player']
        start_pressed_action = start_observation[team]['pressed_action']

        # Get the first step where the controlled player touches the ball
        touch_step = -1
        for step in range(start_step_idx, min(len(observations), start_step_idx + num_steps)):
            now_step = 'step_{}'.format(step)

            player_touch_ball = observations[now_step]['player_touch_ball']
            team_touch_ball = observations[now_step]['team_touch_ball']

            # If the now pressed action of the player is `pass` and the game mode is not
            # `ThrowIn` it means that he is is not performing a pass correctly.
            if self._game_engine.check_pressed_action(team, observations[now_step], action='pass') \
                    and observations[now_step]['game_mode'] != 'e_GameMode_ThrowIn':
                continue

            if team_touch_ball == team_id and player_touch_ball == start_controlled_player:
                touch_step = step
                break

        if touch_step == -1:
            return -1, -1, -1

        for step in range(touch_step + 1, start_step_idx + num_steps):
            now_step = 'step_{}'.format(step)

            player_touch_ball = observations[now_step]['player_touch_ball']
            team_touch_ball = observations[now_step]['team_touch_ball']
            is_in_play = observations[now_step]['is_in_play']

            if not self._game_engine.is_in_play(observations[now_step]):
                return -1, -1, -1

            # Check that after the controlled player touches the ball, he is not
            # the first player that touches the ball again. (If it is, it means
            # that he didn't perform the action)
            if team_touch_ball == team_id and player_touch_ball == start_controlled_player:
                return -1, -1, -1

            if player_touch_ball == -1 and team_touch_ball == -1:
                continue

            if team_touch_ball != team_id:
                if step // STEPS_PER_FRAME - touch_step // STEPS_PER_FRAME + 1 <= 2: # at least 3 frames
                    return -1, -1, -1
                return 0, touch_step, step # bad pass
            else:
                if step // STEPS_PER_FRAME - touch_step // STEPS_PER_FRAME + 1 <= 3: # at least 3 frames
                    return -1, -1, -1
                return 1, touch_step, step # good pass

        return -1, -1, -1
