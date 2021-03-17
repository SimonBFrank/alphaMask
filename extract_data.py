import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from skimage.io import imread
from skimage.color import rgba2rgb


def get_size(root):
        img_height = int(root.find('height').text)
        img_width = int(root.find('size//width').text)
        img_depth = int(root.find('size//depth').text)

        return((img_height, img_width, img_depth))

def get_filename(root):
    """"Gete filename without file extension"""
    filename = root.find('filename').text[:-4]

    return filename

def generate_image(img_path):
        img = imread(img_path)

        if(img.shape[2] == 4):
            img = rgba2rgb(img)

        return img

def get_class(objs):
    label_arr = []
    for obj in objs:
        img_class = str(obj.find('name').text)
        if (img_class == 'without_mask'):
            label_arr.append(0)
        elif (img_class == 'mask_weared_incorrect'):
            label_arr.append(1)
        else:
            label_arr.append(2)

    return label_arr

def get_bbox(objs):
    bbox_arr = []
    for bbox in objs:
        xmin = int(bbox.find('bndbox//xmin').text)
        xmax = int(bbox.find('bndbox//xmax').text)
        ymin = int(bbox.find('bndbox//ymin').text)
        ymax = int(bbox.find('bndbox//ymax').text)

        anchor = (xmin, ymin)
        height = ymax - ymin
        width = xmax - xmin

        bbox_arr.append([anchor, width, height])

    return bbox_arr

def generate_target(annotation_path):
    root = ET.parse(annotation_path)
    objs = root.findall('object')

    filename = get_filename(root)
    bbox_arr = get_bbox(objs)
    labels_arr = get_class(objs)

    return filename, [(bbox, label) for bbox, label in zip(bbox_arr, labels_arr)]


def process_data(img_dir, annotation_dir, processed_img_dir, processed_annotations_dir):
    # Find all the files in annotation and image directories
    img_files = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
    img_annotations = [os.path.join(annotation_dir, path) for path in os.listdir(annotation_dir)]

    # Create new folder for processed data if it does not exist
    if (not os.path.exists(processed_img_dir)):
        os.makedirs(processed_img_dir)
        print('Directory made: '+processed_img_dir)

    if (not os.path.exists(processed_annotations_dir)):
        os.makedirs(processed_annotations_dir)
        print('Directory made: '+processed_annotations_dir)
        
    for image, target in zip(img_files,img_annotations):
        img = generate_image(image)
        filename, target = generate_target(target)

        np.save(os.path.join(processed_annotations_dir, filename+'.npy'), target)
        np.save(os.path.join(processed_img_dir, filename+'.npy'), img)
        
        print('- '+filename+" processed")

if (__name__ == "__main__"):
    print('Starting to extract data')
    
    img_dirs = sys.argv[1]
    annotation_dirs = sys.argv[2]
    processed_img_dir = sys.argv[3]
    processed_annotations_dir = sys.argv[4]

    for img_dir, annotation_dir in zip(img_dirs.split(','), annotation_dirs.split(',')):
        process_data(img_dir, annotation_dir, processed_img_dir, processed_annotations_dir)

    print('Data extraction complete')