import cv2

class Video:
    def __init__(self, video_quality=2):
        self._frame_dim = (1280, 720)
        if video_quality == 2:
            self._fcc = cv2.VideoWriter_fourcc('p', 'n', 'g', ' ')
        else:
            self._fcc = cv2.VideoWriter_fourcc(*'MJPG') # 1
        self._fps = 10
        self._video_format = '.avi'

    def dump_video(self, video_path, frames):
        video_path = video_path + self._video_format
        self._video_writer = cv2.VideoWriter(video_path,
                                             self._fcc,
                                             self._fps,
                                             self._frame_dim)

        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._video_writer.write(frame)

        self._video_writer.release()
