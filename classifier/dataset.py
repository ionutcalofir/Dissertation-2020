import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

def get_transforms(dataset_name):
    if dataset_name == 'train':
        transforms_list = [TransformRescale(),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    else:
        transforms_list = [TransformRescale(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    return transforms.Compose(transforms_list)

class TransformRescale:
    def __init__(self, frame_size=(224, 224)):
        self._frame_size = frame_size

    def __call__(self, frame):
        frame = frame.resize(self._frame_size)
        return frame

class FootballDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_name,
                 data_path,
                 data_cfg):
        np.random.seed(42)
        self._dataset_name = dataset_name
        self._data_path = data_path
        self._data_cfg = data_cfg

        self._videos = []
        self._labels = []
        with open(self._data_cfg, 'r') as f:
            for line in f:
                self._videos.append(line.strip().split()[0])
                self._labels.append(int(line.strip().split()[1]))

        self._transforms = get_transforms(dataset_name)

    def __len__(self):
        return len(self._videos)

    def __getitem__(self, idx):
        for _ in range(10):
            video_path = os.path.join(self._data_path, self._videos[idx])
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print('Something is wrong with {}'.format(video_path))
                idx = np.random.randint(0, len(self))
                continue

            nr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, nr_frames - 1)

            ret, frame = cap.read()

            if not ret:
                print('Something is wrong with {}!'.format(video_path))
                idx = np.random.randint(0, len(self))
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self._transforms(frame)

            label = self._labels[idx]

            break

        return frame, label, video_path
