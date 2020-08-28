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
            logging.info('Failed to decode video {}.'.format(video_path))

        nr_frames = 0
        while (cap.isOpened()):
            ret, frame = cap.read()

            if not ret:
                break

            nr_frames += 1

        cap.release()
        cv2.destroyAllWindows()

        return nr_frames

    def _compute_stats(self, dataset_name):
        self._stats[dataset_name] = {}
        video_recognition_path = os.path.join(self._root_path, dataset_name)

        if dataset_name == 'video_recognition':
            classes = ['no_action', 'pass', 'shot']
        else:
            classes = ['0', '1']

        for c in classes:
            self._stats[dataset_name][c] = {}
            self._stats[dataset_name][c]['videos'] = []

            class_path = os.path.join(video_recognition_path, c)
            videos_name = sorted(os.listdir(class_path))

            for video_name in videos_name:
                video_path = os.path.join(class_path, video_name)

                nr_frames = self._get_stats_video(video_path)
                self._stats[dataset_name][c]['videos'].append((video_name, nr_frames))

        import ipdb; ipdb.set_trace()

    def compute_stats(self):
        self._compute_stats('video_recognition')
        self._compute_stats('expected_goals')
