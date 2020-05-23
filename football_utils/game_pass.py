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

        # Check if the pressed action from the start_observation is the last pressed action for the controlled player
        for step in range(start_step_idx + 1, start_step_idx + num_steps):
            now_step = 'step_{}'.format(step)

            now_controlled_player = observations[now_step][team]['controlled_player']
            now_pressed_action = observations[now_step][team]['pressed_action']

            if now_controlled_player != start_controlled_player:
                break

            if now_pressed_action != 'no_action':
                return -1, -1, -1

        # After the controlled player tries to perform an action check if he is the first person that touches the ball
        start_step_idx_touch = None
        for step in range(start_step_idx + 1, start_step_idx + num_steps):
            now_step = 'step_{}'.format(step)
            now_player_touch_ball = observations[now_step]['player_touch_ball']
            now_team_touch_ball = observations[now_step]['team_touch_ball']

            if not self._game_engine.is_in_play(observations[now_step]):
                return -1, -1, -1

            if now_player_touch_ball == -1 and now_team_touch_ball == -1:
                continue

            if now_player_touch_ball != start_controlled_player or now_team_touch_ball != team_id:
                return -1, -1, -1

            start_step_idx_touch = step
            cont_step = start_step_idx_touch + 1
            while cont_step < start_step_idx + num_steps: # get the last touch
                now_step = 'step_{}'.format(cont_step)
                now_player_touch_ball = observations[now_step]['player_touch_ball']
                now_team_touch_ball = observations[now_step]['team_touch_ball']

                if not self._game_engine.is_in_play(observations[now_step]):
                    break

                if now_player_touch_ball == -1 and now_team_touch_ball == -1:
                    cont_step += 1
                    continue

                if now_player_touch_ball != start_controlled_player or now_team_touch_ball != team_id:
                    break

                start_step_idx_touch = cont_step
                cont_step += 1

            break # The controlled player touched the ball

        if start_step_idx_touch is None:
            return -1, -1, -1

        # Get the first player that touches the ball after the controlled player
        for step in range(start_step_idx_touch, start_step_idx + num_steps):
            now_step = 'step_{}'.format(step)
            now_player_touch_ball = observations[now_step]['player_touch_ball']
            now_team_touch_ball = observations[now_step]['team_touch_ball']

            if not self._game_engine.is_in_play(observations[now_step]):
                return -1, -1, -1

            if now_player_touch_ball == -1 and now_team_touch_ball == -1:
                continue

            if now_player_touch_ball == start_controlled_player and now_team_touch_ball == team_id:
                continue

            if now_team_touch_ball != team_id:
                if step // STEPS_PER_FRAME - start_step_idx_touch // STEPS_PER_FRAME < 1: # at least 2 frames
                    return -1, -1, -1
                return 0, start_step_idx_touch, step # bad pass
            else:
                if step // STEPS_PER_FRAME - start_step_idx_touch // STEPS_PER_FRAME < 1: # at least 2 frames
                    return -1, -1, -1
                return 1, start_step_idx_touch, step # good pass

        return -1, -1, -1
