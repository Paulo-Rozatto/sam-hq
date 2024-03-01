import argparse
import numpy as np
import cv2

from glob import glob
from os import makedirs
from os.path import join, normpath, basename, exists

IN_PATH = normpath("")
OUT_PATH = normpath("")
crop_border = False


def random_crop(image, mask, min_crop_size=256, max_crop_size=512):
    crop_size = np.random.randint(min_crop_size, max_crop_size + 1)
    max_x = image.shape[1] - crop_size
    max_y = image.shape[0] - crop_size

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    image = image[y: y + crop_size, x: x + crop_size]
    mask = mask[y: y + crop_size, x: x + crop_size]

    return image, mask


def random_crop_border(image, mask, min_crop_size=256, max_crop_size=512):
    crop_size = np.random.randint(min_crop_size, max_crop_size + 1)

    if crop_size == 1024:
        return image, mask

    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnt, key=cv2.contourArea)
    poly = cv2.approxPolyDP(cnt, 0.0005 * cv2.arcLength(cnt, True), True)
    print(len(poly))
    poly = poly.reshape(-1, 2)

    idx = np.random.randint(0, len(poly))
    x, y = poly[idx]

    length = int(crop_size / 2)

    top = y - length
    bottom = y + length

    left = x - length
    right = x + length

    if (top < 0):
        offset = -top
        top += offset
        bottom += offset
    elif (bottom > image.shape[0]):
        offset = bottom - image.shape[0]
        top -= offset
        bottom -= offset

    if (left < 0):
        offset = -left
        left += offset
        right += offset
    elif (right > image.shape[0]):
        offset = right - image.shape[0]
        left -= offset
        right -= offset

    image = image[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    return image, mask


def scale_up(image, mask, size=1024):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
    return image, mask


def transform(image, mask):
    is_valid_crop = False

    crop_fn = random_crop_border if crop_border else random_crop
    while not is_valid_crop:
        img, msk = crop_fn(image, mask)
        is_valid_crop = (cv2.countNonZero(msk) / msk.size) > 0.1

    img, msk = scale_up(img, msk, 1024)
    return img, msk


def main():
    paths_images = glob(join(IN_PATH, "train_images", "*.jpg"))
    paths_masks = glob(join(IN_PATH, "train_masks", "*.jpg"))

    out_image_dir = join(OUT_PATH, "train_images")
    out_mask_dir = join(OUT_PATH, "train_masks")

    # if not exists(out_image_dir):
    #     makedirs(out_image_dir)
    # if not exists(out_mask_dir):
    #     makedirs(out_mask_dir)

    for image_path, mask_path in zip(paths_images, paths_masks):
        name = basename(image_path)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        new_image_path = join(out_image_dir, name)
        new_mask_path = join(out_mask_dir, name)
        cv2.imwrite(new_image_path, image)
        cv2.imwrite(new_mask_path, mask)

        for i in range(2):
            image, mask = transform(image, mask)
            new_image_path = join(out_image_dir, f"{i}_{name}")
            new_mask_path = join(out_mask_dir, f"{i}_{name}")
            cv2.imwrite(new_image_path, image)
            cv2.imwrite(new_mask_path, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random crop and resize bean leaf dataset"
    )

    parser.add_argument(
        "-b", "--border", action="store_true"
    )
    parser.add_argument("-i", "--in_path", type=str)
    parser.add_argument("-o", "--out_path", type=str)

    args = parser.parse_args()
    print(args.in_path)

    if args.in_path is not None:
        IN_PATH = normpath(args.in_path)
    if args.out_path is not None:
        OUT_PATH = normpath(args.out_path)

    main()
