import cv2
import xml.etree.cElementTree as ET
import os
import matplotlib.pyplot as plt

def find_faces():
    NUMBER_FILES = 3000
    incorrect_faces_filename = []
    incorrect_faces_imgs = []
    incorrect_faces_bbox = []
    for subdir, dirs, files in os.walk('./../scene_data'):
        if (len(incorrect_faces_bbox) > NUMBER_FILES):
            break
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".jpg") and filepath.find('Copy') == -1:
                img = plt.imread(filepath)
                img = img[:,:,::-1]
                incorrect_faces_filename.append(file)
                incorrect_faces_imgs.append(img)
                incorrect_faces_bbox.append([])
                if (len(incorrect_faces_bbox) > NUMBER_FILES):
                    break

    return incorrect_faces_filename, incorrect_faces_imgs, incorrect_faces_bbox

def create_xml(filename, img_shape, bbox, save_loc):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = str(filename)

    doc = ET.SubElement(root, "size")
    ET.SubElement(doc, "width").text = str(img_shape[0])
    ET.SubElement(doc, "height").text = str(img_shape[1])
    ET.SubElement(doc, "depth").text = str(3)

    tree = ET.ElementTree(root)
    tree.write(save_loc+filename+'.xml')

def export_data():
    if (os.path.isdir('./../scene_selected/') == False):
        os.mkdir('./../scene_selected/')
        os.mkdir('./../scene_selected/imgs/')
        os.mkdir('./../scene_selected/labels/')
    
    incorrect_faces_filename, incorrect_faces_imgs, incorrect_faces_bbox = find_faces()
    for filename, img, bbox in zip(incorrect_faces_filename, incorrect_faces_imgs, incorrect_faces_bbox):
        create_xml(filename, img.shape, bbox, './../scene_selected/labels/')
        cv2.imwrite('./../scene_selected/imgs/'+filename, img)

if (__name__ == "__main__"):
    export_data()