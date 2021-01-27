import os
import xml.etree.ElementTree as ET
import skimage.io as imageIO
import skimage.color as transformColor
import tensorflow as tf
import numpy as np

class MaskDataset:
    def __init__(self, img_path, annotation_path):
        self.img_files = [os.path.join(img_path, path) for path in os.listdir(img_path)]
        self.img_annotations = [os.path.join(annotation_path, path) for path in os.listdir(annotation_path)]

    def get_size(self, idx):
        root = ET.parse(self.img_annotations[idx])

        img_height = int(root.find('height').text)
        img_width = int(root.find('size//width').text)
        img_depth = int(root.find('size//depth').text)

        return((img_height, img_width, img_depth))

    def get_filename(self, idx):
        filename = self.img_files[idx]

        return filename

    def get_image(self, img_path):
        img = imageIO.imread(img_path)

        return img

    def get_class(self, objs):
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

    def get_bbox(self, objs):
        bbox_arr = []
        for bbox in objs:
            xmin = int(bbox.find('bndbox//xmin').text)
            xmax = int(bbox.find('bndbox//xmax').text)
            ymin = int(bbox.find('bndbox//ymin').text)
            ymax = int(bbox.find('bndbox//ymax').text)

            bbox_arr.append([xmin, xmax, ymin, ymax])

        return bbox_arr

    def generate_target(self, annotation_path):
        objs = ET.parse(annotation_path).findall('object')

        bbox_arr = self.get_bbox(objs)
        labels_arr = self.get_class(objs)

        return [(bbox, label) for bbox, label in zip(bbox_arr, labels_arr)]

    def get_item(self, idx):
        img_path = self.img_files[idx]
        label_path = self.img_annotations[idx]

        img = self.get_image(img_path)
        target = self.generate_target(label_path)

        return img, target

    def generate_data(self):
        X = []
        y = []
        for i in range(len(self.img_annotations)):
            img, target = self.get_item(i)
            X.append(tf.convert_to_tensor(img))
            y.append(target)

        return np.array(X), y