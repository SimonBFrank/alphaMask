import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

#NOTES
#1. Debug image boundary conditions, some images display as black
#2. Complete Sequence object for DataGenerator
#3. Add helper to translate bounding box annotations for YOLO format
#4. Add brightness transformation to augmentation scheme

class Reflection(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bbox):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img = img[:, ::-1, :]
            bbox[:, [0, 2]] += 2 * (img_center[[0, 2]] - bbox[:, [0, 2]])
            box_width = np.abs(bbox[:, 0] - bbox[:, 2])
            bbox[:, 0] -= box_width
            bbox[:, 2] += box_width

        return img, bbox

class Scale(object):
    def __init__(self, scale = 0.2, fixed_aspect = True):
        self.scale = (max(-1, -scale), scale)
        self.fixed_aspect = fixed_aspect

    def __call__(self, img, bbox):
        img_shape = img.shape
        if self.fixed_aspect:
            h_scale = 1 + random.uniform(*self.scale)
            v_scale = h_scale
        else:
            h_scale = 1 + random.uniform(*self.scale)
            v_scale = 1 + random.uniform(*self.scale)

        print(img)
        img = cv2.resize(img, None, fx = h_scale, fy = v_scale ,)

        bbox[:, :4] *= [h_scale, v_scale, h_scale, v_scale]

        canvas = np.zeros(img_shape, dtype = np.uint8)
        h_bound = int(min(h_scale, 1) * img_shape[1])
        v_bound = int(min(v_scale, 1) * img_shape[0])
        canvas[:v_bound, :h_bound, :] = img[:v_bound, :h_bound, :]
        img = canvas
        bbox = trim_bbox(bbox, [0, 0, img_shape[1], img_shape[0]], 0.25)

        return img, bbox

class Translation(object):
    def __init__(self, translate = 0.2, fixed_aspect = False):
        self.translate = (-translate, translate)
        self.fixed_aspect = fixed_aspect

    def __call__(self, img, bbox):
        img_shape = img.shape
        
        if self.fixed_aspect:
            h_translation = random.uniform(*self.translate)
            v_translation = h_translation
        else:
            h_translation = random.uniform(*self.translate)
            v_translation = random.uniform(*self.translate)
        
        canvas = np.zeros(img_shape, dtype = np.uint8)
        corner_x = int(h_translation * img_shape[1])
        corner_y = int(v_translation * img_shape[0])

        # may require cv2 optimization (cv2.warpAffine)
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]
        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        bbox[:,:4] += [corner_x, corner_y, corner_x, corner_y]
        bbox = trim_bbox(bbox, [0, 0, img_shape[1], img_shape[0]], 0.25)

        return img, bbox

class Rotation(object):
    def __init__(self, angle = 10):
        self.angle = (-angle, angle)

    def __call__(self, img, bbox):
        angle = random.uniform(*self.angle)
        image_shape = img.shape
        h_center = image_shape[1] // 2
        v_center = image_shape[0] // 2

        transform = cv2.getRotationMatrix2D((h_center, v_center), angle, 1.0)
        cos = np.abs(transform[0, 0])
        sin = np.abs(transform[0, 1])

        transform_width = int((img.shape[0] * sin) + (img.shape[1] * cos))
        transform_height = int((img.shape[0] * cos) + (img.shape[1] * sin))
        transform[0, 2] += (transform_width / 2) - h_center
        transform[1, 2] += (transform_height / 2) - v_center

        img = cv2.warpAffine(img, transform, (transform_width, transform_height))

        corners = query_bbox(bbox)
        corners = np.hstack((corners, bbox[:, 4:]))
        corners[:, :8] = rotate_bbox(corners[:, :8], angle, h_center, v_center, image_shape[0], image_shape[1])
        rotated_bbox = fill_bbox(corners)

        bbox = rotated_bbox
        bbox = trim_bbox(bbox, [0, 0, image_shape[1], image_shape[0]], 0.25)

        return img, bbox

####### HELPERS #######
def format_bbox(bbox):
    formatted_bbox = np.empty((bbox.shape[0], 5))
    for i in range(bbox.shape[0]):
        ul_corner = np.array(bbox[:, 0][i][0])
        bbox_width = bbox[:, 0][i][1]
        bbox_height = bbox[:, 0][i][2]
        lr_corner = np.array([ul_corner[0] + bbox_width, ul_corner[1] + bbox_height])
        formatted_bbox[i] = np.hstack((ul_corner, lr_corner, bbox[:, 1][i]))
        
    return formatted_bbox

def display_bbox(img, bbox):
    rect_img = img.copy()
    color_dict = {0: [255, 0, 0], 1: [255, 165, 0], 2: [0, 255, 0]}
    for i in range(bbox.shape[0]):
        pt1 = int(bbox[i][0]), int(bbox[i][1])
        pt2 = int(bbox[i][2]), int(bbox[i][3])
        cat = int(bbox[i][4])
        rect_img = cv2.rectangle(rect_img, pt1, pt2, color_dict[cat], int(max(img.shape[:2])/200))

    return rect_img

def trim_bbox(bbox, img_coords, threshold):
    area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    x_min = np.maximum(bbox[:, 0], img_coords[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], img_coords[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], img_coords[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], img_coords[3]).reshape(-1, 1)
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    reshape_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    area_diff = ((area - reshape_area)/area)
    mask = (area_diff < (1 - threshold)).astype(int)
    bbox = bbox[mask == 1]

    return bbox

def query_bbox(bbox):
    width = (bbox[:, 2] - bbox[:, 0]).reshape(-1, 1)
    height = (bbox[:, 3] - bbox[:, 1]).reshape(-1, 1)

    x1 = bbox[:, 0].reshape(-1, 1)
    y1 = bbox[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bbox[:, 2].reshape(-1, 1)
    y4 = bbox[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
    return corners

def rotate_bbox(corners, angle, h_center, v_center, height, width):
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

    transform = cv2.getRotationMatrix2D((h_center, v_center), angle, 1.0)
    cos = np.abs(transform[0, 0])
    sin = np.abs(transform[0, 1])
    transform_width = int((img.shape[0] * sin) + (img.shape[1] * cos))
    transform_height = int((img.shape[0] * cos) + (img.shape[1] * sin))
    transform[0, 2] += (transform_width / 2) - h_center
    transform[1, 2] += (transform_height / 2) - v_center

    rotated = np.dot(transform, corners.T).T
    rotated = rotated.reshape(-1, 8)
    return rotated

def fill_bbox(corners):
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]
    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)
    
    bbox = np.hstack((xmin, ymin, xmax, ymax, corners[:,8:]))
    
    return bbox

##########
img = np.load('processed/images/maksssksksss0.npy', allow_pickle=True)
targets = np.load('processed/annotations/maksssksksss0.npy', allow_pickle=True)

bbox = format_bbox(targets)

horizontal_flip = Reflection(1)
img, bbox = horizontal_flip(img, bbox)

scale = Scale(1)
img, bbox = scale(img, bbox)

#translation = Translation(0.5)
#img, bbox = translation(img, bbox)

rotation = Rotation(20)
img, bbox = rotation(img, bbox)
plt.imshow(display_bbox(img, bbox))
plt.show()
