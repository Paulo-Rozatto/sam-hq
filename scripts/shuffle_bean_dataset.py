from random import shuffle
from glob import glob
from math import ceil
from os import path, listdir, makedirs
from shutil import copy

in_path = '/home/fourier/paulo/fine-tune-301-330--396-410/'
out_path = '/home/fourier/paulo/sam-hq/train/data/fine-tune-301-330--396-410/'

test_images_path = path.join(out_path, "test_images")
test_masks_path = path.join(out_path, "test_masks")

train_images_path = path.join(out_path, "train_images")
train_masks_path = path.join(out_path, "train_masks")

def make_dir(dir_path):
    if not path.exists(dir_path):
        makedirs(dir_path)

def copy_files(src, dst):
    files = glob(path.join(src, "*"))

    for file in files:
        file_dst = path.join(dst, path.basename(file))
        print(file)
        copy(file, dst)

make_dir(out_path)
make_dir(test_images_path)
make_dir(test_masks_path)
make_dir(train_images_path)
make_dir(train_masks_path)


folders = listdir(in_path)
shuffle(folders)

with open("shuffle-301-330--396-410.txt", "w") as save_file:
    str_shuffle = ",".join(folders)
    save_file.write(str_shuffle)


test_size = ceil(len(folders) * 0.2)

print("--------------------")
print("TEST")
print("--------------------")
for i in range(0, test_size):
    image_path = path.join(in_path, folders[i], "images")
    copy_files(image_path, test_images_path)
    

    mask_path = path.join(in_path, folders[i], "masks")
    copy_files(mask_path, test_masks_path)

print("--------------------")
print("TRAIN")
print("--------------------")
for i in range(test_size, len(folders)):
    image_path = path.join(in_path, folders[i], "images")
    copy_files(image_path, train_images_path)

    mask_path = path.join(in_path, folders[i], "masks")
    copy_files(mask_path, train_masks_path)
    
