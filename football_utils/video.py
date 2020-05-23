import os
import cv2
import math
import subprocess
from football_utils.football_constants import *

class Video:
    def __init__(self, downscale_videos, video_quality=2):
        self._frame_dim = (1280, 720)
        if video_quality == 2:
            self._fcc = cv2.VideoWriter_fourcc('p', 'n', 'g', ' ')
        else:
            self._fcc = cv2.VideoWriter_fourcc(*'MJPG') # 1
        self._fps = 10
        self._video_format = '.avi'
        self._downscale_videos = downscale_videos

        if self._downscale_videos:
            factor = min(self._frame_dim) / SHORT_EDGE_SIZE
            self._frame_dim = (math.floor(self._frame_dim[0] / factor),
                               math.floor(self._frame_dim[1] / factor))

    def dump_video(self, video_path, frames):
        if self._downscale_videos:
            video_path += '_tmp'
        video_path = video_path + self._video_format
        self._video_writer = cv2.VideoWriter(video_path,
                                             self._fcc,
                                             self._fps,
                                             self._frame_dim)

        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (self._frame_dim[0], self._frame_dim[1]))
            self._video_writer.write(frame)

        self._video_writer.release()

        if self._downscale_videos:
            # video_path[:-8] - _tmp.avi
            new_video_path = video_path[:-8] + self._video_format
            self.downscale_video(video_path, new_video_path)
            os.remove(video_path)

    def downscale_video(self, inname, outname):
        """
        Original script here:
        https://github.com/facebookresearch/video-nonlocal-net/blob/master/process_data/kinetics/downscale_video_joblib.py
        """
        status = False
        inname = '"{}"'.format(inname)
        outname = '"{}"'.format(outname)
        command = "ffmpeg  -loglevel panic -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -q:v 1 -c:a copy {}".format(inname, outname)

        status = os.path.exists(outname.strip('"'))
        if status:
            print('File {} already exists and will be replaced!'.format(outname.strip('"')))
            os.remove(outname.strip('"'))

        try:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            print(status, 'Error: {}'.format(err.output))

        status = os.path.exists(outname.strip('"'))
        if not status:
            raise Exception('Error: Something went wrong!')
