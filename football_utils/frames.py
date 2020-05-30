import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Frames:
    def get_frame(self, frame_path):
        with open(frame_path, 'rb') as f:
            frame = pickle.load(f)
        return frame

    def get_frames_path(self, dump_path):
        frames_path = os.path.join(dump_path, 'frames')
        frames_path = [os.path.join(frames_path, frame)
                       for frame in sorted(os.listdir(frames_path), key=lambda x: int(x.split('_')[-1][:-4]))]
        return frames_path

    def plot_point(self, frame, point):
        fig, ax = plt.subplots()
        ax.imshow(frame)
        ax.scatter(point[0], point[1], c='r', s=20)
        plt.show()
