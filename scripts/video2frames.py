from copy import deepcopy
from email.mime import base
import cv2
import os

from rembg.bg import remove
from PIL import Image
import tqdm

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def remove_bg(filepath):
    image = Image.open(filepath)
    result = remove(image)
    if filepath[-3:] == 'png':
        img = result.convert("RGBA")
    else:
        img = result.convert("RGB")
    img.save(filepath)


basedir = "data/venus-rough-1-2-w-background"
video_name = 'venus_rough_video_1.mp4'

video = cv2.VideoCapture(os.path.join(basedir, video_name))

os.makedirs(os.path.join(basedir, 'images'), exist_ok=True)
os.makedirs(os.path.join(basedir, 'images_jpg'), exist_ok=True)

print("Step 1: Extracting frames")
i = 0
while video.isOpened():
    ret, frame = video.read()
    if ret == False:
        break

    # Resize frame
    # frame = cv2.resize(frame, (800, 800))

    # Remove background
    # frame = remove_bg(frame)

    # _ = cv2.imwrite(os.path.join(basedir, 'source/', video_name[:-4]) + f'_{i}.png', frame, )
    _ = cv2.imwrite(os.path.join(basedir, 'images/' + f'r_{i}.png'), frame, )
    # _ = cv2.imwrite(os.path.join(basedir, 'source_jpg/', video_name[:-4]) + f'_{i}.jpg', frame, )
    _ = cv2.imwrite(os.path.join(basedir, 'images_jpg/' + f'r_{i}.jpg'), frame, )

    i += 1

print("Step 2: Removing background")
for j in tqdm.tqdm(range(i)):
    # remove_bg(os.path.join(basedir, 'source/', video_name[:-4]) + f'_{j}.png')
    remove_bg(os.path.join(basedir, 'images/' + f'r_{j}.png'))
    remove_bg(os.path.join(basedir, 'images_jpg/' + f'r_{j}.jpg'))

print("DONE")
