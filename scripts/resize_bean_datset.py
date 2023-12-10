import os
import cv2
import numpy as np

from os import path
from sys import argv
from glob import glob
from bs4 import BeautifulSoup as bs

IN_PATH = ""
OUT_PATH = ""

OUT_SIZE = 1024
HEIGHT = 4624

def readXml(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = "".join(lines)
        return bs(lines, "lxml")

def get_points_from_xml(xml_path, tag = "points"):
    points = readXml(xml_path).find(tag).strings
    points = [float(x) for x in list(points) if x != '\n']
    return points

def points_to_contour(points, scale = 1):
   if len(points) % 2 != 0:
       print("Error: points should have an even length")
       return None
   
   half =  int(len(points) / 2)
   points = np.array(points) * scale

   return points.reshape((half, 1, 2)).astype(np.int32)

def get_contour_center(contour):
    M = cv2.moments(contour)

    if M['m00'] == 0:
        print("Warning: m00 == 0")
        return (0,0)

    center_x = int(M['m10']/M['m00'])
    center_y = int(M['m01']/M['m00'])
    return np.array([center_x, center_y])
        
if __name__ == "__main__":
    if len(argv) > 1:
        IN_PATH = path.normpath(argv[1])
    if len(argv) > 2:
        OUT_PATH = path.normpath(argv[2])

    if not path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    dirs_names = os.listdir(IN_PATH)
    dirs_names.sort()

    for dir_name in dirs_names:
        print(dir_name)
        dir_path = path.join(IN_PATH, dir_name)

        images_paths = glob(path.join(dir_path, "*.jpg"))
        images_paths.sort()

        out_dir = path.join(OUT_PATH, dir_name)
        if not path.exists(out_dir):
            os.mkdir(out_dir)

        out_images_dir = path.join(out_dir, "images")
        if not path.exists(out_images_dir):
            os.mkdir(out_images_dir)

        out_masks_dir = path.join(out_dir, "masks")
        if not path.exists(out_masks_dir):
            os.mkdir(out_masks_dir)

        for image_path in images_paths:
            image_name = path.basename(image_path)
            print(image_name)

            leaf_path = path.join(dir_path, "leaf", image_name.replace("jpg", "xml"))

            points = get_points_from_xml(leaf_path)
            contour = points_to_contour(points, HEIGHT)

            x, y, width, height = cv2.boundingRect(contour)
            size = max(width, height)

            offset = 100
            top = np.array([x, y]) - offset
            bot = np.array([x + size, y + size]) + offset

            image = cv2.imread(image_path)

            mask = np.zeros(image.shape[:2], np.uint8)
            mask = cv2.drawContours(mask, [contour], 0, (255), -1)

            # adjust bottom if its out of range
            dim = [image.shape[1], image.shape[0]]
            for i in range(2):
                if bot[i] > dim[i]:
                    aux = bot[i] - dim[i]
                    top[i] -= aux
                    bot[i] = dim[i]
            
            # adjust top if its out of range
            for i in range(2):
                if top[i] < 0:
                    bot[i] -= top[i] 
                    top[i] = 0
            
            test = [top[0] < 0, top[1] < 0, bot[0] > dim[0], bot[1] > dim[1]]
            if any(test):
                print("------ Erro ------")
                print("top:", top)
                print("bot:", bot)
                continue

            image = image[top[1]:bot[1], top[0]:bot[0]]
            image = cv2.resize(image, (OUT_SIZE, OUT_SIZE))
            save_path = path.join(out_images_dir, image_name)
            cv2.imwrite(save_path, image)
            
            mask = mask[top[1]:bot[1], top[0]:bot[0]]
            mask = cv2.resize(mask, (OUT_SIZE, OUT_SIZE))
            save_path = path.join(out_masks_dir, image_name)
            cv2.imwrite(save_path, mask)

        #     cv2.imshow("image", mask)
        #     cv2.waitKey(0)
        # # closing all open windows 
        # cv2.destroyAllWindows() 
