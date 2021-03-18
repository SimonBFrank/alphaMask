import numpy as np
import os
import random
from PIL import Image
from augmentation import *

random.seed(0)

def export_data(training_data, validation_data, img_export_dir, label_export_dir, data_increase=5):
    transforms = Sequence([Brightness(), Reflection(1), Scale(), Translation(), Rotation()], probs=0.25)

    image_number = 1
    for sample in training_data:
        target = sample[0]
        im = sample[1]
        
        if (im.dtype != 'uint8'):
            im = np.array(im * 255).astype('uint8')

        target = format_bbox(target)
        target = format_yolo(target, im)

        im = Image.fromarray(im)

        img_filename = "training" + str(image_number) + ".jpg"
        im.save(img_export_dir + "training/" + img_filename)


        label_filename = "training" + str(image_number) + ".txt"
        np.savetxt(label_export_dir+"training/"+label_filename, target, fmt='%1.6f')

        image_number += 1

    for sample in validation_data:
        target = sample[0]
        im = sample[1]
        
        if (im.dtype != 'uint8'):
            im = np.array(im * 255).astype('uint8')

        target = format_bbox(target)
        target = format_yolo(target, im)

        im = Image.fromarray(im)

        img_filename = "validation" + str(image_number) + ".jpg"
        im.save(img_export_dir + "validation/" + img_filename)


        label_filename = "validation" + str(image_number) + ".txt"
        np.savetxt(label_export_dir+"validation/"+label_filename, target, fmt='%1.6f')

        image_number += 1


if (__name__ == '__main__'):
    img_dir = '../processed/images/'
    label_dir = '../processed/annotations/'
    img_export_dir = './data/images/'
    label_export_dir = './data/labels/'

    imgs = [np.load(img_dir+ img_path) for img_path in os.listdir(img_dir)]
    labels = [np.load(label_dir+label_path, allow_pickle=True) for label_path in os.listdir(label_dir)]

    data = [(label, img) for img, label in zip(imgs, labels)]
    random.shuffle(data)

    data_size = len(data)
    training_size = 0.7

    training_data = data[:int(data_size*training_size)]
    validation_data = data[int(data_size*training_size):]

    export_data(training_data, validation_data, img_export_dir, label_export_dir)