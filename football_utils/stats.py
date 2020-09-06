import os
import cv2
import matplotlib.pyplot as plt
from absl import logging


class Stats():
    def __init__(self, root_path):
        self._root_path = root_path
        self._stats = {}

    def _get_stats_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            nr_frames = 0
            logging.info('Failed to decode video {}.'.format(video_path))
        else:
            nr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()
        cv2.destroyAllWindows()

        return nr_frames

    def _compute_stats(self, dataset_name):
        print('Process dataset: {}'.format(dataset_name))

        self._stats[dataset_name] = {}
        video_recognition_path = os.path.join(self._root_path, dataset_name)

        if dataset_name == 'video_recognition':
            classes = ['no_action', 'pass', 'shot']
        else:
            classes = ['0', '1']

        for c in classes:
            print('Process class: {}'.format(c))

            self._stats[dataset_name][c] = {}
            self._stats[dataset_name][c]['videos'] = []

            class_path = os.path.join(video_recognition_path, c)
            videos_name = sorted(os.listdir(class_path))

            for idx, video_name in enumerate(videos_name):
                print('{} video {}/{}'.format(c, idx, len(videos_name)))
                video_path = os.path.join(class_path, video_name)

                nr_frames = self._get_stats_video(video_path)
                self._stats[dataset_name][c]['videos'].append((video_name, nr_frames))

    def _show_stats(self, dataset_name):
        print('Showing stats for {}'.format(dataset_name))
        print('-----')

        if dataset_name == 'video_recognition':
            classes = ['no_action', 'pass', 'shot']
        else:
            classes = ['0', '1']
        games = set()
        total_frames = 0
        total_videos = []
        avg_frames = []

        for c in classes:
            no_frames = 0
            no_videos = 0
            for video in self._stats[dataset_name][c]['videos']:
                if video[1] == 0: # 0 frames
                    continue

                no_frames += video[1]
                no_videos += 1

                game_name = '_'.join(video[0].split('_')[:2])
                games.add(game_name)

            total_frames += no_frames
            avg_frames.append(no_frames / no_videos)
            total_videos.append(no_videos)

        fig, ax = plt.subplots()
        ax.bar(classes, avg_frames)
        ax.set_title('Avg frames per video for each class')

        fig, ax = plt.subplots()
        ax.bar(classes, total_videos)
        ax.set_title('Number of video for each class')

        for i in range(len(classes)):
            print('Total videos for {}: {}'.format(classes[i], total_videos[i]))
            print('Avg frames for {}: {}'.format(classes[i], avg_frames[i]))
            print()

        print('Total matches: {}'.format(len(games)))
        print('Total number of frames: {}'.format(total_frames))
        print('Total number of seconds: {}'.format(total_frames // 10))
        print()

        plt.show()

    def compute_stats(self):
        self._compute_stats('video_recognition')
        self._compute_stats('expected_goals')

        self._show_stats('video_recognition')
        self._show_stats('expected_goals')
