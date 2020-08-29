import json
import os
from joblib import Parallel, delayed
from absl import logging

from football_utils.football_constants import *
from football_utils.observations import Observations
from football_utils.frames import Frames
from football_utils.game_engine import GameEngine
from football_utils.game_pass import GamePass
from football_utils.game_goals import GameGoals
from football_utils.video import Video

class GameGeneration:
    def __init__(self,
                 dataset_path,
                 dataset_output_path,
                 downscale_videos):

        self._dataset_path = dataset_path
        self._dataset_output_path = dataset_output_path
        self._downscale_videos = downscale_videos

        self._sliding_window_frames_stride = 5
        self._sliding_window_frames_lengths = [10, 15, 20]

        self._observations = Observations()
        self._frames = Frames()
        self._game_engine = GameEngine()
        self._pass = GamePass()
        self._goals = GameGoals()
        self._video = Video(self._downscale_videos)

    def generate(self):
        os.makedirs(self._dataset_output_path, exist_ok=True)

        actions = ['pass', 'shot', 'expected_goals']
        for dump_nr, dump_name in enumerate(sorted(os.listdir(self._dataset_path))):
            game_path = os.path.join(self._dataset_output_path, dump_name)
            try:
                os.makedirs(game_path)
            except:
                logging.info('Game {} already exists and it will be skipped!'.format(dump_name))
                continue

            ground_truth_videos_path = os.path.join(game_path, 'gt_videos')
            os.makedirs(ground_truth_videos_path)
            ground_truth_videos_path_observations = ground_truth_videos_path + '_observations'
            os.makedirs(ground_truth_videos_path_observations)

            logging.info('Preprocess dump {} ({}/{})'.format(dump_name, dump_nr + 1, len(os.listdir(self._dataset_path))))
            dump_path = os.path.join(self._dataset_path, dump_name)

            frames_path = self._frames.get_frames_path(dump_path)
            observations = self._observations.get_observations(dump_path)

            action_frames = {k : [] for k in actions}
            information_action_frames = {k : {} for k in actions}

            for step_idx in range(len(observations)):
                start_observation = self._observations.get_observation(step_idx, observations)

                for action in actions:
                    action_type, start_frame_idx, end_frame_idx = \
                        self._game_engine.get_frame_action(step_idx, start_observation, observations, min(DATASET_GENERATION_FRAMES_WINDOW, len(observations) - 1 - step_idx),
                                                           action=action, pass_object=self._pass, goals_object=self._goals)

                    if action_type == -1:
                            continue

                    if action in ['pass', 'shot']:
                        start_frame_idx = max(0, start_frame_idx // STEPS_PER_FRAME - 2) # -2 frame back for more information
                        end_frame_idx = min(end_frame_idx // STEPS_PER_FRAME + 3, len(observations) // STEPS_PER_FRAME - 1) # +3 frame back for more information
                    elif action == 'expected_goals':
                        end_frame_idx = start_frame_idx // STEPS_PER_FRAME # it stops right when the player performs the shot (start_frame_idx is not put there by mistake)
                        start_frame_idx = max(0, start_frame_idx // STEPS_PER_FRAME - 20) # -20 frames back before the player performs the shot

                    observations_frames = {}
                    for step_frame in range(start_frame_idx * STEPS_PER_FRAME, end_frame_idx * STEPS_PER_FRAME):
                        step_frame_name = 'step_{}'.format(step_frame)
                        observations_frames[step_frame_name] = self._observations.get_observation(step_frame, observations)

                    if not (len(action_frames[action]) > 0 and start_frame_idx == action_frames[action][-1][0]):
                        action_frames[action].append((start_frame_idx, end_frame_idx, observations_frames))

                        information_action_frames[action][start_frame_idx] = {}
                        information_action_frames[action][start_frame_idx]['is_goal'] = False
                        information_action_frames[action][start_frame_idx]['is_good_pass'] = False
                        information_action_frames[action][start_frame_idx]['end_frame_idx'] = end_frame_idx

                        if action == 'shot' and action_type == 1:
                            information_action_frames[action][start_frame_idx]['is_goal'] = True
                        if action == 'pass' and action_type == 1:
                            information_action_frames[action][start_frame_idx]['is_good_pass'] = True

            def save_videos(i, action_frame, cls, num_examples):
                print('Preprocess class {} frames {}/{}'.format(cls, i + 1, num_examples))
                base_name = 'video_{}_{}'.format(cls, action_frame[0])

                video_path = os.path.join(ground_truth_videos_path, base_name)
                frames = [self._frames.get_frame(frames_path[frame]) for frame in range(action_frame[0], action_frame[1])]
                self._video.dump_video(video_path, frames)

                observations_path = os.path.join(ground_truth_videos_path_observations, base_name)
                self._observations.dump_observations(observations_path, action_frame[2])

            for cls in action_frames:
                Parallel(n_jobs=-2)(delayed(save_videos)(i, action_frame, cls, len(action_frames[cls])) for i, action_frame in enumerate(action_frames[cls]))

            root_sliding_window_videos_path = os.path.join(game_path, 'sliding_window_videos')

            for sliding_window_frames_length in self._sliding_window_frames_lengths:
                print('Preprocess sliding window frames length {}'.format(sliding_window_frames_length))
                sliding_window_videos_path = os.path.join(root_sliding_window_videos_path, 'sliding_window_videos')
                sliding_window_videos_information_path = sliding_window_videos_path + '_information'

                sliding_window_videos_path = sliding_window_videos_path + '_length_{}'.format(sliding_window_frames_length)
                sliding_window_videos_information_path = sliding_window_videos_information_path + '_length_{}'.format(sliding_window_frames_length)

                os.makedirs(sliding_window_videos_path)
                os.makedirs(sliding_window_videos_information_path)

                sliding_window_config = []
                sliding_window_frames = []
                information_sliding_window_frames = {}
                for idx in range(0, len(frames_path), self._sliding_window_frames_stride):
                    start_frame_idx = idx
                    end_frame_idx = min(idx + sliding_window_frames_length, len(frames_path))

                    sliding_window_frames.append((start_frame_idx, end_frame_idx))
                    sliding_window_video_name = 'sliding_window_{}_{}'.format(start_frame_idx, len(sliding_window_frames))
                    sliding_window_config.append(sliding_window_video_name)

                    information_sliding_window_frames[sliding_window_video_name] = {}
                    information_sliding_window_frames[sliding_window_video_name]['start_frame_idx'] = start_frame_idx
                    information_sliding_window_frames[sliding_window_video_name]['end_frame_idx'] = end_frame_idx

                    if idx + sliding_window_frames_length >= len(frames_path):
                        break

                with open(sliding_window_videos_information_path + '/sliding_window_videos_information.json', 'w') as f:
                    json.dump(information_sliding_window_frames, f)
                with open(sliding_window_videos_information_path + '/test.csv', 'w') as f:
                    for sliding_window_video_name in sliding_window_config:
                        f.write('sliding_window_videos/{}.avi {}\n'.format(sliding_window_video_name, -1))

                def save_videos_sliding_window(i, sliding_window_frame, num_examples):
                    print('Preprocess frames {}/{}'.format(i + 1, num_examples))
                    base_name = 'sliding_window_{}_{}'.format(sliding_window_frame[0], i + 1)

                    video_path = os.path.join(sliding_window_videos_path, base_name)
                    frames = [self._frames.get_frame(frames_path[frame]) for frame in range(sliding_window_frame[0], sliding_window_frame[1])]
                    self._video.dump_video(video_path, frames)

                Parallel(n_jobs=-2)(delayed(save_videos_sliding_window)(i, sliding_window_frame, len(sliding_window_frames)) for i, sliding_window_frame in enumerate(sliding_window_frames))
