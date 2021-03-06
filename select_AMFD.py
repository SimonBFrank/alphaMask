import cv2
import xml.etree.cElementTree as ET
import os
import numpy as np

def find_faces():
    NUMBER_FILES = 2400
    
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    absent_faces_filename = []
    absent_faces_imgs = []
    absent_faces_bbox = []
    for subdir, dirs, files in os.walk('./../AMFD'):
        if (len(absent_faces_bbox) >NUMBER_FILES):
            break
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".png") and filepath.find('Copy') == -1 and filepath.find('Mask') == -1:
                img = cv2.imread(filepath)
                new_height = int(400 * np.random.normal(1, 0.1, (1,))[0])
                new_width = int(500 * np.random.normal(1, 0.1, (1,))[0])
                img = cv2.resize(img, (new_width, new_height))
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray_img,
                    scaleFactor=1.3,
                    minNeighbors=10
                )
                if (len(faces) == 1):
                    absent_faces_filename.append(file[:-4]+'.jpg')
                    absent_faces_imgs.append(img)
                    absent_faces_bbox.append(faces)
                    if (len(absent_faces_bbox) >NUMBER_FILES):
                        break

    return absent_faces_filename, absent_faces_imgs, absent_faces_bbox

def create_xml(filename, img_shape, bbox, save_loc):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = str(filename)

    doc = ET.SubElement(root, "size")
    ET.SubElement(doc, "width").text = str(img_shape[0])
    ET.SubElement(doc, "height").text = str(img_shape[1])
    ET.SubElement(doc, "depth").text = str(3)

    for box in bbox:
        xmin = box[0]
        ymin = box[1]
        xmax = box[0] + box[2]
        ymax = box[1] + box[3]
        doc = ET.SubElement(root, "object")
        ET.SubElement(doc, "name").text = "without_mask"
        bbox = ET.SubElement(doc, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)
        


    tree = ET.ElementTree(root)
    tree.write(save_loc+filename+'.xml')

def export_data():
    if (os.path.isdir('./../amfd_selected/') == False):
        os.mkdir('./../amfd_selected/')
        os.mkdir('./../amfd_selected/imgs/')
        os.mkdir('./../amfd_selected/labels/')
    correct_faces_filename, correct_faces_imgs, correct_faces_bbox = find_faces()
    for filename, img, bbox in zip(correct_faces_filename, correct_faces_imgs, correct_faces_bbox):
        create_xml(filename, img.shape, bbox, './../amfd_selected/labels/')
        cv2.imwrite('./../amfd_selected/imgs/'+filename, img)

if (__name__ == "__main__"):
    export_data()