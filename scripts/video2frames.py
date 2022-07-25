import os
import tqdm

import cv2
from rembg.bg import remove
from PIL import Image, ImageFile
from utils import filter_all

ImageFile.LOAD_TRUNCATED_IMAGES = True


def remove_bg(filepath):
    image = Image.open(filepath)
    result = remove(image)
    if filepath[-3:] == 'png':
        img = result.convert("RGBA")
    else:
        img = result.convert("RGB")
    img.save(filepath)


basedir = "data/venus-rough-1-texture-tests-1"
video_name = 'venus_rough_video_1.mp4'

video = cv2.VideoCapture(os.path.join(basedir, video_name))

os.makedirs(os.path.join(basedir, 'images'), exist_ok=True)
os.makedirs(os.path.join(basedir, 'images_jpg'), exist_ok=True)

print("Step 1: Extracting frames")
i = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    _ = cv2.imwrite(os.path.join(basedir, 'images/' + f'r_{i}.png'), frame, )
    _ = cv2.imwrite(os.path.join(basedir, 'images_jpg/' + f'r_{i}.jpg'), frame, )

    i += 1

print("Step 2: Texturizing background")
filter_all(os.path.join(basedir, 'images'))

print("Step 3: Removing background")
for j in tqdm.tqdm(range(i)):
    remove_bg(os.path.join(basedir, 'images_filtered/' + f'r_{j}.png'))
    # remove_bg(os.path.join(basedir, 'images_jpg/' + f'r_{j}.jpg'))

print("DONE")
