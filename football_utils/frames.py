import os
import numpy as np
from PIL import Image

class Frames:
    def get_frame(self, frame_path):
        frame = np.array(Image.open(frame_path))
        return frame

    def get_frames_path(self, dump_path):
        frames_path = os.path.join(dump_path, 'frames')
        frames_path = [os.path.join(frames_path, frame)
                       for frame in sorted(os.listdir(frames_path), key=lambda x: int(x.split('_')[-1][:-4]))]
        return frames_path
