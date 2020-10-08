import os
import cv2
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

# video_path = 'images/expected_goals/283.gif'
# output_path = 'images/expected_goals'
# name = '283'

video_path = 'images/video_recognition/shot_1988.gif'
output_path = 'images/video_recognition'
name = 'shot_1988'

output_path = os.path.join(output_path, name)
os.makedirs(output_path, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0
frames = []
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # frame.save('{}/frame_{}.png'.format(output_path, frame_idx))
    frames.append(frame)

if len(frames) % 2 == 1:
    frames = frames[:-1]

for idx, frame in enumerate(frames):
    frame.save('{}/{}_frame_{}.png'.format(output_path, name, idx + 1))

# fig = plt.figure(figsize=(30., 15.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(2, len(frames) // 2, 2),  # creates 2xlen(frames) grid of axes
#                  axes_pad=0.05,  # pad between axes in inch.
#                  )

# for ax, frame in zip(grid, frames):
#     # Iterating over the grid returns the Axes.
#     ax.imshow(frame)
#     ax.axis('off')

# plt.savefig('{}/{}_grid.png'.format(output_path, name), bbox_inches='tight')
# plt.close()
print('Done!')
