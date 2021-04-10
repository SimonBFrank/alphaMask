import cv2
import xml.etree.cElementTree as ET
import os

def find_faces():
    NUMBER_FILES = 3000
    
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    incorrect_faces_filename = []
    incorrect_faces_imgs = []
    incorrect_faces_bbox = []
    for subdir, dirs, files in os.walk('./../IMFD'):
        if (len(incorrect_faces_bbox) >NUMBER_FILES):
            break
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".jpg") and filepath.find('Copy') == -1:
                img = cv2.imread(filepath)
                gray_img = gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray_img,
                    scaleFactor=1.3,
                    minNeighbors=10
                )
                if (len(faces) == 1):
                    incorrect_faces_filename.append(file)
                    incorrect_faces_imgs.append(img)
                    incorrect_faces_bbox.append(faces)
                    if (len(incorrect_faces_bbox) >NUMBER_FILES):
                        break

    return incorrect_faces_filename, incorrect_faces_imgs, incorrect_faces_bbox

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
        ET.SubElement(doc, "name").text = "mask_weared_incorrect"
        bbox = ET.SubElement(doc, "bndbox")
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymax").text = str(ymax)
        ET.SubElement(bbox, "xmax").text = str(xmax)


    tree = ET.ElementTree(root)
    tree.write(save_loc+filename+'.xml')

def export_data():
    if (os.path.isdir('./../imfd_selected/') == False):
        os.mkdir('./../imfd_selected/')
        os.mkdir('./../imfd_selected/imgs/')
        os.mkdir('./../imfd_selected/labels/')
    incorrect_faces_filename, incorrect_faces_imgs, incorrect_faces_bbox = find_faces()
    for filename, img, bbox in zip(incorrect_faces_filename, incorrect_faces_imgs, incorrect_faces_bbox):
        create_xml(filename, img.shape, bbox, './../imfd_selected/labels/')
        cv2.imwrite('./../imfd_selected/imgs/'+filename, img)

if (__name__ == "__main__"):
    export_data()