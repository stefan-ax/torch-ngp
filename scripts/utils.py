from PIL import Image, ImageFilter
import numpy as np
from glob import glob
import os
from tqdm.auto import tqdm


def emboss(image):
    imageEmboss = image.filter(ImageFilter.EMBOSS)
    return imageEmboss


def sharpen(image, times=0):
    imageSharpen = image.filter(ImageFilter.SHARPEN)

    for _ in range(times):
        imageSharpen = imageSharpen.filter(ImageFilter.SHARPEN)

    return imageSharpen


def filter_image(image_path, original_weight, emboss_weight, sharpen_weight):
    # assert(original_weight + emboss_weight + sharpen_weight == 1.0)

    # Get original image
    image = Image.open(image_path)
    image_np = np.array(image)

    # Emboss
    imageEmboss = emboss(image)
    imageEmboss_np = np.array(imageEmboss)

    # Sharpen
    imageSharpen = sharpen(image, times=3)
    imageSharpen_np = np.array(imageSharpen)

    # Add them together
    final_image_np = original_weight * image_np + emboss_weight * imageEmboss_np + sharpen_weight * imageSharpen_np
    final_image = Image.fromarray(final_image_np.round().astype('uint8'), mode="RGB")

    return final_image


def filter_all(folder_path):
    # Get all the path to images in the folder
    image_paths = glob(folder_path + "/*.png")

    print("Filtering images: ", image_paths)

    # Create a new images_filtered directory
    filtered_folder = folder_path + "_filtered"
    os.makedirs(filtered_folder, exist_ok=True)

    # For each filepath, apply the filter
    for path in tqdm(image_paths):
        Image_filtered = filter_image(path, 0.1, 0.6, 0.3)
        Image_filtered.save(os.path.join(filtered_folder, os.path.basename(path)))

    print(f"Saved to: {filtered_folder}")
