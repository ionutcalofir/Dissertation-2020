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

        # Get the first step where the controlled player touches the ball
        touch_step = -1
        for step in range(start_step_idx, min(len(observations), start_step_idx + num_steps)):
            now_step = 'step_{}'.format(step)

            player_touch_ball = observations[now_step]['player_touch_ball']
            team_touch_ball = observations[now_step]['team_touch_ball']

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
            is_goal_scored = observations[now_step]['is_goal_scored']

            # Check that after the controlled player touches the ball, he is not
            # the first player that touches the ball again. (If it is, it means
            # that he didn't perform the action)
            if team_touch_ball == team_id and player_touch_ball == start_controlled_player:
                return -1, -1, -1

            if (team_touch_ball != -1 and player_touch_ball != -1) \
                    or (not is_in_play):
                if is_in_play:
                    # Check the case where the keeper or a player touches the ball but after
                    # it's still goal (deflection).
                    for future_step in range(step + 1, start_step_idx + num_steps):
                        future_now_step = 'step_{}'.format(future_step)

                        future_player_touch_ball = observations[future_now_step]['player_touch_ball']
                        future_team_touch_ball = observations[future_now_step]['team_touch_ball']
                        future_is_goal_scored = observations[future_now_step]['is_goal_scored']

                        if future_is_goal_scored:
                            return 1, touch_step, future_step

                        if future_player_touch_ball == -1 and future_team_touch_ball == -1:
                            continue
                        else:
                            break

                    return 0, touch_step, step
                else:
                    if is_goal_scored:
                        return 1, touch_step, step
                    else:
                        return 0, touch_step, step

        return -1, -1, -1
