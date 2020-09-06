import os
import cv2
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

video_path = 'videos/pass.avi'
output_path = 'images/pass'
name = 'pass'

# video_path = 'videos/shot.avi'
# output_path = 'images/shot'
# name = 'shot'

# video_path = 'videos/no_action.avi'
# output_path = 'images/no_action'
# name = 'no_action'

if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

cap = cv2.VideoCapture(video_path)
frame_idx = 0
frames = []
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = Image.fromarray(frame).resize((224, 224))
    frame = Image.fromarray(frame)

    if name == 'pass':
        frame = frame.crop((110, 50, 290, 215)) # left, top, right, bottom
    elif name == 'shot':
        frame = frame.crop((240, 40, frame.size[0], 190)) # left, top, right, bottom
    elif name == 'no_action':
        frame = frame.crop((70, 30, 210, 100)) # left, top, right, bottom

    frame.save('{}/frame_{}.png'.format(output_path, frame_idx))
    frames.append(frame)

if len(frames) % 2 == 1:
    frames = frames[:-1]

fig = plt.figure(figsize=(22., 7.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, len(frames) // 2),  # creates 2xlen(frames) grid of axes
                 axes_pad=0.05,  # pad between axes in inch.
                 )

for ax, frame in zip(grid, frames):
    # Iterating over the grid returns the Axes.
    ax.imshow(frame)
    ax.axis('off')

plt.savefig('{}/{}_grid.png'.format(output_path, name), bbox_inches='tight')
plt.close()
print('Done!')
