import numpy as np
import cv2 as cv
from math import ceil
from random import randint
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time
from os import listdir

from skimage.feature import hog

from sklearn.svm import LinearSVC

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

TRAIN_DATA_PATH = "./antrenare/"
VALIDATION_DATA_PATH = "./validare/"

FACE_LEN = 72
CELL_LEN = 8
BLOCK_CELL_LEN = 3

IOU_NMS_THRESHOLD = 0.05
IOU_VALIDATION_THRESHOLD = 0.3

# stats and IO

def get_train_data():

    def _img_name(i):

        name = f"{i}"
        while len(name) != 4:
            name = "0" + name

        return f"pic_{name}.jpg"

    IMG_PERFOLDER_CNT = 1100

    all_imgs = []
    all_desc = []

    img_cnt = IMG_PERFOLDER_CNT
    for name in ["bart", "homer", "lisa", "marge"]:

        imgs = []        
        processed_desc = []

        with open(f"{TRAIN_DATA_PATH}{name}.txt", "r") as f:
            desc = f.read().split("\n")

        j = 0
        for i in range(img_cnt):
            
            imgn = _img_name(i)
            imgs.append(cv.imread(f"{TRAIN_DATA_PATH}{name}/{imgn}"))

            processed_desc.append([])

            while j < len(desc) and desc[j][:12] == imgn:

                desc[j] = desc[j].split()
                processed_desc[i].append(((int(desc[j][1]), int(desc[j][2])), (int(desc[j][3]), int(desc[j][4])), desc[j][5]))

                j += 1
        
        all_imgs += imgs
        all_desc += processed_desc

    return all_imgs, all_desc

def get_validation_data():

    with open(f"{VALIDATION_DATA_PATH}simpsons_validare.txt", "r") as f:
        desc = f.read().split("\n")

    opened = {}

    i = 0
    for d in desc:
        
        d = d.split()
        
        if d[0] not in opened.keys():
            opened[d[0]] = (cv.imread(f"{VALIDATION_DATA_PATH}simpsons_validare/{d[0]}"), i, [])
            i += 1

        opened[d[0]][2].append(((int(d[1]), int(d[2])), (int(d[3]), int(d[4])), d[5]))

    imgs = [v[0] for v in opened.values()]
    desc = [v[2] for v in opened.values()]

    return imgs, desc

def get_test_data(path):

    img_names = [n for n in listdir(path)]
    imgs = [cv.imread(f"{path}{img_n}") for img_n in img_names]

    return imgs, img_names

def show_predictions(imgs, predictions):

    def _get_y_bycnt(img):

        cnt = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                if img[i][j][0] < 80 and img[i][j][1] > 100 and img[i][j][2] > 150:
                    cnt += 1

        return cnt / (img.shape[0] * img.shape[1])

    for i in range(len(imgs)):
        
        img = imgs[i]
        for pred in predictions[i]:

            x1, y1, x2, y2, score, y = pred[0], pred[1], pred[2], pred[3], pred[4], pred[5]

            cv.line(img, (x1, y1), (x2, y1), color=(255, 0, 0))
            cv.line(img, (x1, y1), (x1, y2), color=(255, 0, 0))
            cv.line(img, (x2, y1), (x2, y2), color=(255, 0, 0))
            cv.line(img, (x1, y2), (x2, y2), color=(255, 0, 0))

        cv.imshow(f"prediction idx {i}", img)
        cv.waitKey(0)

def get_precision_recall_task1(predictions, desc):

    true_positives = 0
    false_positives = 0
    total_faces_cnt = 0

    for i in range(len(predictions)):

        total_faces_cnt += len(desc[i])

        matched = [False for _ in range(len(predictions[i]))]

        for real_face in desc[i]:

            x1, y1 = real_face[0][0], real_face[0][1]
            x2, y2 = real_face[1][0], real_face[1][1]

            for i_pred in range(len(predictions[i])):

                if matched[i_pred]:
                    continue

                pred = predictions[i][i_pred]

                x1_, y1_, x2_, y2_ = pred[0], pred[1], pred[2], pred[3]
                if iou(x1, y1, x2, y2, x1_, y1_, x2_, y2_) > IOU_VALIDATION_THRESHOLD:
                    
                    true_positives += 1
                    matched[i_pred] = True
                    break

        for i_pred in range(len(predictions[i])):
            if not matched[i_pred]:
                false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_faces_cnt

    print(f"precision: {precision}, recall {recall}")
    return precision, recall

def save_predictions_task1(img_predictions, img_names):

    TASK1_PREDICTION_PATH = "./predictii/task1"

    detections = []
    scores = []
    file_names = []

    for idx in range(len(img_names)):

        for p in img_predictions[idx]:

            detections.append([p[0], p[1], p[2], p[3]])
            scores.append(p[6])
            file_names.append(img_names[idx])

    detections = np.array(detections)
    scores = np.array(scores)
    file_names = np.array(file_names)

    np.save(f"{TASK1_PREDICTION_PATH}detections_all_faces", detections)
    np.save(f"{TASK1_PREDICTION_PATH}scores_all_faces", scores)
    np.save(f"{TASK1_PREDICTION_PATH}file_names_all_faces", file_names)

# training and feature extraction

def get_pos_neg_hog(imgs, desc, neg_factor=1):

    def _preprocess(img_):

        img_ = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
        img_ = cv.resize(img_, (FACE_LEN, FACE_LEN))

        return img_

    pos_cnt = 0
    neg_cnt = 0

    pos_hog = []
    neg_hog = []

    # redundant, but increases readability
    pos_names = []

    for i in range(len(imgs)):
        img = imgs[i]

        for j in range(len(desc[i])):

            x1, y1 = desc[i][j][0]
            x2, y2 = desc[i][j][1]
            name = desc[i][j][2]
            
            face = _preprocess(img[y1: y2, x1: x2, :])

            pos_hog.append(hog(face, pixels_per_cell=(CELL_LEN, CELL_LEN), cells_per_block=(BLOCK_CELL_LEN, BLOCK_CELL_LEN)))
            pos_names.append(name)

            pos_cnt += 1

    neg_cnt = ceil(pos_cnt * neg_factor)

    def _get_coord4neg(img, check_against):
        
        while True:

            y_ = randint(0, img.shape[0] - FACE_LEN)
            x_ = randint(0, img.shape[1] - FACE_LEN)

            ok = True
            for j in range(len(check_against)):

                x1, y1 = check_against[j][0]
                x2, y2 = check_against[j][1]

                if (x1 <= x_ <= x2 and y1 <= y_ <= y2) or (x1 <= x_ + FACE_LEN <= x2 and y1 <= y_ + FACE_LEN <= y2):
                    ok = False
                    break

            if ok:
                break

        return x_, y_

    for i in range(len(imgs)):
        img = imgs[i]

        for _ in range(ceil(neg_factor)):
            
            x_, y_ = _get_coord4neg(img, desc[i])

            img_processed = _preprocess(img[y_: y_ + FACE_LEN, x_: x_ + FACE_LEN, :])
            neg_hog.append(hog(img_processed, pixels_per_cell=(CELL_LEN, CELL_LEN), cells_per_block=(BLOCK_CELL_LEN, BLOCK_CELL_LEN)))

    return pos_hog, pos_names, neg_hog[:neg_cnt]

def get_traindata_cnn(imgs, desc, neg_factor=1):

    def _preprocess(img_):

        img_ = cv.resize(img_, (FACE_LEN, FACE_LEN))

        return img_

    pos_cnt = 0
    neg_cnt = 0

    pos_samples = []
    neg_samples = []

    # redundant, but increases readability
    pos_names = []

    for i in range(len(imgs)):
        img = imgs[i]

        for j in range(len(desc[i])):

            x1, y1 = desc[i][j][0]
            x2, y2 = desc[i][j][1]
            name = desc[i][j][2]
            
            face = _preprocess(img[y1: y2, x1: x2, :])

            pos_samples.append(face)
            pos_names.append(name)

            pos_cnt += 1

    neg_cnt = ceil(pos_cnt * neg_factor)

    def _get_coord4neg(img, check_against):
        
        while True:

            y_ = randint(0, img.shape[0] - FACE_LEN)
            x_ = randint(0, img.shape[1] - FACE_LEN)

            ok = True
            for j in range(len(check_against)):

                x1, y1 = check_against[j][0]
                x2, y2 = check_against[j][1]

                if (x1 <= x_ <= x2 and y1 <= y_ <= y2) or (x1 <= x_ + FACE_LEN <= x2 and y1 <= y_ + FACE_LEN <= y2):
                    ok = False
                    break

            if ok:
                break

        return x_, y_

    for i in range(len(imgs)):
        img = imgs[i]

        for _ in range(ceil(neg_factor)):
            
            x_, y_ = _get_coord4neg(img, desc[i])

            img_processed = _preprocess(img[y_: y_ + FACE_LEN, x_: x_ + FACE_LEN, :])
            neg_samples.append(img_processed)

    return pos_samples, pos_names, neg_samples[:neg_cnt]

def train_svm_task1(pos_hog, neg_hog):

    # code partially adapted from CAVA course, lab10

    best_accuracy = 0
    best_model = None

    train_data = np.array(pos_hog + neg_hog)
    train_labels = np.array([True for _ in range(len(pos_hog))] + [False for _ in range(len(neg_hog))])

    cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1]#, 10 ** 0]
    for c in cs:

        model = LinearSVC(C=c)
        model.fit(train_data, train_labels)
        acc = model.score(train_data, train_labels)

        print(f"svm(c = {c}) accuracy {acc}")

        if acc > best_accuracy:

            best_accuracy = acc
            best_model = deepcopy(model)

    pickle.dump(best_model, open("best_svm_task1", "wb+"))

    return best_model

def train_nn_task1(pos_hog, neg_hog):

    hog_feature_cnt = pos_hog[0].shape[0]

    train_data = np.array(pos_hog + neg_hog)
    train_labels = np.array([1 for _ in range(len(pos_hog))] + 
                            [0 for _ in range(len(neg_hog))])

    train_labels = tf.one_hot(train_labels, depth=2)
    
    model = Sequential()
    model.add(Dense(512, input_dim=hog_feature_cnt, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x=train_data, y=train_labels, epochs=20, batch_size=64)

    save_model(model, "best_dense_nn_hog_task1")
    return model

def train_cnn_task1(pos_samples, neg_samples):

    train_data = np.array(pos_samples + neg_samples)
    train_labels = np.array([1 for _ in range(len(pos_samples))] + 
                            [0 for _ in range(len(neg_samples))])

    train_labels = tf.one_hot(train_labels, depth=2)
    
    model = Sequential()
    
    model.add(Conv2D(32, input_shape=(FACE_LEN, FACE_LEN, 3), kernel_size=(4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Dropout(0.6))

    model.add(Conv2D(64, kernel_size=(4, 4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(256))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.4))

    model.add(Dense(64))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(Dense(16))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # TODO change to at least 5
    model.fit(x=train_data, y=train_labels, epochs=5, batch_size=32)

    save_model(model, "best_cnn_task1")
    return model

# utilitary functions

def iou(x1, y1, x2, y2, x1_, y1_, x2_, y2_):

    assert(x1 < x2 and y1 < y2 and x1_ < x2_ and y1_ < y2_)
    
    # NOTE: due to the big number of all possible cases
    #       instead of treating each one separately
    #       I opted for a simpler (but slower) method of calculating areas
    #       due to the relatively small sizes taken into consideration
    #       this approach should suffice this particular use case

    # fast response if 0
    if (x1 >= x2_ and y1 >= y2_) or (x1_ >= x2 and y1_ >= y2):
        return 0

    a1 = (x2 - x1) * (y2 - y1)
    a2 = (x2_ - x1_) * (y2_ - y1_)

    intersection = 0

    xdif = min(x1, x1_)
    ydif = min(y1, y1_)

    x1 -= xdif
    x2 -= xdif
    x1_ -= xdif
    x2_ -= xdif

    y1 -= ydif
    y2 -= ydif
    y1_ -= ydif
    y2_ -= ydif

    xmax = max(x2, x2_)
    ymax = max(y2, y2_)

    mat = [[0 for _ in range(ymax)] for _ in range(xmax)]
    
    for i in range(x1, x2):
        for j in range(y1, y2):

            mat[i][j] += 1

    for i in range(x1_, x2_):
        for j in range(y1_, y2_):

            mat[i][j] += 1

    for i in range(xmax):
        for j in range(ymax):

            if mat[i][j] == 2:
                intersection += 1

    return intersection / (a1 + a2 - intersection)

def non_maximum_suppression(scores):
    
    scores = sorted(scores, key = lambda t: t[4])

    filtered = []

    while len(scores) > 0:

        biggest = scores[-1]
        scores = scores[:-1]

        if biggest[4] <= 0:

            if len(filtered) == 0:
                filtered.append(biggest)

            return filtered

        i = 0
        while i < len(scores):

            if iou(biggest[0], biggest[1], biggest[2], biggest[3], 
                    scores[i][0], scores[i][1], scores[i][2], scores[i][3]) > IOU_NMS_THRESHOLD:

                scores = scores[:i] + scores[i + 1:]

            else:
                i += 1

        filtered.append(biggest)

    return filtered

# task 1 specific

# NOTE: after about 20 hours of work
#       I realised my window was sliding only on diagonal :))))
#       but because I had already implemented cnn at this point
#       I did not bother to change 
#       the prediction function for the other (first) two methods

def predictions_smv_task1(model: LinearSVC, imgs):

    # resize image
    # equivalent to sliding windows of different sizes and shapes (rectangular / square)
    def _resized_img(img):
        
        #for resize_dim in [0.5, 0.75, 1, 1.25, 1.5]:
         #   for yx_ratio in [(2, 1), (1.7, 1), (1.5, 1), (1.25, 1), (1.1, 1), (1, 1), (1, 1.1), (1, 1.25)]:

        for resize_dim in [0.5, 0.75, 1, 1.25, 1.5, 2]:
           for yx_ratio in [(2, 1), (1.7, 1), (1.5, 1), (1.25, 1), (1, 1)]:

                yield resize_dim, yx_ratio, cv.resize(img, (int(resize_dim * img.shape[0] // yx_ratio[0]), int(resize_dim * img.shape[1] // yx_ratio[1])))

    # get coordinates for the (not necessarily square) window in the original image
    # corresponding to the square window in the reshaped image
    def _get_org_coords(resize_dim, yx_ratio, y1, x1):
        return int(y1 * yx_ratio[0] // resize_dim), int(x1 * yx_ratio[1] // resize_dim)

    scores = []

    _cnt = 0
    for initial_img in imgs:

        print(f"prediction for idx {_cnt}")
        _cnt += 1

        scores.append([])

        # TODO: check for unwanted shifts
        #       due to imprecise divisions when resizing

        for resize_dim, yx_ratio, img in _resized_img(initial_img):

            # I have the resized images, 
            # all chunks will be squares but they will cover the rectangular cases anyways
            # due to yx_ratio used

            ymax, xmax, _ = img.shape
            if ymax < FACE_LEN or xmax < FACE_LEN:
                continue

            x1, y1 = 0, 0
            while x1 + FACE_LEN <= xmax and y1 + FACE_LEN <= ymax:

                img_hog, _, _ = get_pos_neg_hog([img], [[((x1, y1), (x1 + FACE_LEN, y1 + FACE_LEN), None)]], 0)

                y1_org, x1_org = _get_org_coords(resize_dim, yx_ratio, y1, x1)
                y2_org, x2_org = _get_org_coords(resize_dim, yx_ratio, y1 + FACE_LEN, x1 + FACE_LEN)

                scores[-1].append([x1_org, y1_org, x2_org, y2_org, model.decision_function(np.array(img_hog))[0]])

                x1 += CELL_LEN
                y1 += CELL_LEN

        assert(len(scores[-1]) > 0)

        # NOTE: the sliding window was a square, but it was applied on 
        #       a wide rande of resized versions of the initial image
        #       to cover as much cases as possible
        #       the non maximum suppression will be applied on all those 
        #       different-sized chunks

        scores[-1] = non_maximum_suppression(scores[-1])

    return scores

def predictions_hog_w_dense_nn_task1(model: Sequential, imgs):

    # resize image
    # equivalent to sliding windows of different sizes and shapes (rectangular / square)
    def _resized_img(img):
        
        #for resize_dim in [0.5, 0.75, 1, 1.25, 1.5]:
         #   for yx_ratio in [(2, 1), (1.7, 1), (1.5, 1), (1.25, 1), (1.1, 1), (1, 1), (1, 1.1), (1, 1.25)]:

        for resize_dim in [0.75, 1, 1.25, 1.5, 2]:
           for yx_ratio in [(2, 1), (1.7, 1), (1.5, 1), (1.25, 1), (1, 1)]:

                yield resize_dim, yx_ratio, cv.resize(img, (int(resize_dim * img.shape[0] // yx_ratio[0]), int(resize_dim * img.shape[1] // yx_ratio[1])))

    # get coordinates for the (not necessarily square) window in the original image
    # corresponding to the square window in the reshaped image
    def _get_org_coords(resize_dim, yx_ratio, y1, x1):
        return int(y1 * yx_ratio[0] // resize_dim), int(x1 * yx_ratio[1] // resize_dim)

    scores = []

    _cnt = 0
    for initial_img in imgs:

        print(f"prediction for idx {_cnt}")
        _cnt += 1

        scores.append([])

        # TODO: check for unwanted shifts
        #       due to imprecise divisions when resizing

        for resize_dim, yx_ratio, img in _resized_img(initial_img):

            # I have the resized images, 
            # all chunks will be squares but they will cover the rectangular cases anyways
            # due to yx_ratio used

            ymax, xmax, _ = img.shape
            if ymax < FACE_LEN or xmax < FACE_LEN:
                continue

            x1, y1 = 0, 0
            while x1 + FACE_LEN <= xmax and y1 + FACE_LEN <= ymax:

                img_hog, _, _ = get_pos_neg_hog([img], [[((x1, y1), (x1 + FACE_LEN, y1 + FACE_LEN), None)]], 0)

                y1_org, x1_org = _get_org_coords(resize_dim, yx_ratio, y1, x1)
                y2_org, x2_org = _get_org_coords(resize_dim, yx_ratio, y1 + FACE_LEN, x1 + FACE_LEN)

                prediction = model.predict(np.array(img_hog))
                scores[-1].append([x1_org, y1_org, x2_org, y2_org, np.argmax(prediction, axis=1)])

                x1 += CELL_LEN
                y1 += CELL_LEN

        assert(len(scores[-1]) > 0)

        # NOTE: the sliding window was a square, but it was applied on 
        #       a wide rande of resized versions of the initial image
        #       to cover as much cases as possible
        #       the non maximum suppression will be applied on all those 
        #       different-sized chunks

        scores[-1] = non_maximum_suppression(scores[-1])

    return scores

def predictions_cnn_task1(model: Sequential, imgs):
    
    # resize image
    # equivalent to sliding windows of different sizes and shapes (rectangular / square)
    def _resized_img(img):
        
        #for resize_dim in [0.5, 0.75, 1, 1.25, 1.5]:
         #   for yx_ratio in [(2, 1), (1.7, 1), (1.5, 1), (1.25, 1), (1.1, 1), (1, 1), (1, 1.1), (1, 1.25)]:

        for resize_dim in [0.35, 0.5, 0.75, 1, 1.25, 1.5, 2]:
           for yx_ratio in [(2.2, 1), (1.9, 1), (1.5, 1), (1.25, 1), (1, 1)]:

                yield resize_dim, yx_ratio, cv.resize(img, (int(resize_dim * img.shape[0] // yx_ratio[0]), int(resize_dim * img.shape[1] // yx_ratio[1])))

    # get coordinates for the (not necessarily square) window in the original image
    # corresponding to the square window in the reshaped image
    def _get_org_coords(resize_dim, yx_ratio, y1, x1):
        return int(y1 * yx_ratio[0] // resize_dim), int(x1 * yx_ratio[1] // resize_dim)
    
    # method responsible for filtering out 
    # some candidates that are most likely NOT faces
    def _get_y_matrix(img):

        # pad matrix with +1 to avoid out of bounds access further on
        # due to mul/div operations on some indexes

        y_mat = [[0 for _ in range(img.shape[1] + 1)] for _ in range(img.shape[0] + 1)]
        r_mat = [[0 for _ in range(img.shape[1] + 1)] for _ in range(img.shape[0] + 1)]
        
        for i in range(img.shape[0]):
            
            if img[i][0][0] < 80 and img[i][0][1] > 100 and img[i][0][2] > 150:
                y_mat[i][0] += 1

            if img[i][0][0] > 180 and img[i][0][1] > 180 and img[i][0][2] > 180:
                r_mat[i][0] += 1

        for j in range(img.shape[1]):
                
            if img[0][j][0] < 80 and img[0][j][1] > 100 and img[0][j][2] > 150:
                y_mat[0][j] += 1

            if img[0][j][0] > 180 and img[0][j][1] > 180 and img[0][j][2] > 180:
                r_mat[0][j] += 1
        
        for i in range(1, img.shape[0]):
            for j in range(1, img.shape[1]):
                
                if img[i][j][0] < 80 and img[i][j][1] > 100 and img[i][j][2] > 150:
                    y_mat[i][j] += 1

                if img[i][j][0] > 180 and img[i][j][1] > 180 and img[i][j][2] > 180:
                    r_mat[i][j] += 1

                y_mat[i][j] += y_mat[i - 1][j] + y_mat[i][j - 1] - y_mat[i - 1][j - 1]
                r_mat[i][j] += r_mat[i - 1][j] + r_mat[i][j - 1] - r_mat[i - 1][j - 1]
        
        # assign values to padding cells

        for i in range(img.shape[0]):
            
            y_mat[i][img.shape[1]] = y_mat[i][img.shape[1] - 1]
            r_mat[i][img.shape[1]] = r_mat[i][img.shape[1] - 1]

        for j in range(img.shape[1]):
            
            y_mat[img.shape[0]][j] = y_mat[img.shape[0] - 1][j]
            r_mat[img.shape[0]][j] = r_mat[img.shape[0] - 1][j]

        y_mat[img.shape[0]][img.shape[1]] = y_mat[img.shape[0] - 1][img.shape[1] - 1]
        r_mat[img.shape[0]][img.shape[1]] = r_mat[img.shape[0] - 1][img.shape[1] - 1]

        return y_mat, r_mat

    def _get_y(x1, y1, x2, y2, y_mat, r_mat, use_eyes=True):

        def _logpow(x, exp):

            if exp <= 16:
                return x ** exp

            if exp & 1 == 0:
                return _logpow(x * x, exp // 2)

            return x * _logpow(x * x, exp // 2)

        val = y_mat[y2][x2] + y_mat[y1][x1] - y_mat[y2][x1] - y_mat[y1][x2]
        
        if use_eyes:

            R_BASE = 1.001
            val *= _logpow(R_BASE, r_mat[y2][x2] + r_mat[y1][x1] - r_mat[y2][x1] - r_mat[y1][x2])

        return val / ((x2 - x1) * (y2 - y1))

    #model.save("tempmodel")
    #model = load_model("tempmodel", compile=False)

    scores = []

    _cnt = 0
    for initial_img in imgs:

        print(f"prediction for idx {_cnt}")
        _cnt += 1

        scores.append([])

        Y_THRESH = 0.6

        backup = None
        backup_y = -1

        y_mat, r_mat = _get_y_matrix(initial_img)

        for resize_dim, yx_ratio, img in _resized_img(initial_img):
            
            # I have the resized images, 
            # all chunks will be squares but they will cover the rectangular cases anyways
            # due to yx_ratio used

            ymax, xmax, _ = img.shape
            if ymax < FACE_LEN or xmax < FACE_LEN:
                continue

            # filtering images that do not surpass yellow threeshold
            # and prepare the others for batch prediction

            for x1 in range(0, xmax + 1 - FACE_LEN, CELL_LEN):
                for y1 in range(0, ymax + 1 - FACE_LEN, CELL_LEN):

                    img_preprocessed = img[y1: y1 + FACE_LEN, x1: x1 + FACE_LEN]

                    y1_org, x1_org = _get_org_coords(resize_dim, yx_ratio, x1, y1)
                    y2_org, x2_org = _get_org_coords(resize_dim, yx_ratio, x1 + FACE_LEN, y1 + FACE_LEN)

                    assert(y2_org <= initial_img.shape[0])
                    assert(x2_org <= initial_img.shape[1])

                    y = _get_y(x1_org, y1_org, x2_org, y2_org, y_mat, r_mat, True)

                    if y > Y_THRESH:
                        scores[-1].append([x1_org, y1_org, x2_org, y2_org, img_preprocessed, y, None])

                        # from now on, no need for backup
                        backup_y = 100000000
                        backup = None

                    elif y > backup_y: 

                        backup_y = y
                        backup = [[x1_org, y1_org, x2_org, y2_org, img_preprocessed, y, None]]

        if len(scores[-1]) > 0:

            print(f"idx {len(scores) - 1}, to predict cnt {len(scores[-1])}")

            batch = [s_[4] for s_ in scores[-1]]
            predicted = model.predict(np.array(batch))

            assert(len(predicted) == len(scores[-1]))
        
            for pr_idx in range(len(predicted)):

                pr = predicted[pr_idx]

                assert(pr[0] + pr[1] > 0.99)

                scores[-1][pr_idx][6] = pr[1]

                if pr[1] < pr[0]:
                    pr = -1
                else:
                    pr = pr[1]

                scores[-1][pr_idx][4] = pr

        elif len(scores[-1]) == 0:

            # at least, best worst case
            # this approach helps in the few cases when 
            # in the image even the "main" face does not surpass the y threshold
            # but at the same time we do not want to lower the y threshold

            pr = model(tf.convert_to_tensor([backup[0][4]]))[0]

            assert(pr[0] + pr[1] > 0.99)

            backup[0][6] = pr[1]

            if pr[1] < pr[0]:
                pr = -1
            else:
                pr = pr[1]

            backup[0][4] = pr
            scores[-1] = backup

        # NOTE: the sliding window was a square, but it was applied on 
        #       a wide rande of resized versions of the initial image
        #       to cover as much cases as possible
        #       the non maximum suppression will be applied on all those 
        #       different-sized chunks

        scores[-1] = non_maximum_suppression(scores[-1])
        assert(scores[-1] is scores[len(scores) - 1])

    return scores

def task1_hog_svm():

    imgs, desc = get_train_data()

    pos_hog, _, neg_hog = get_pos_neg_hog(imgs, desc, 5)
    model = train_svm_task1(pos_hog, neg_hog)

    imgs, desc = get_validation_data()

    # TODO: remove
    # imgs = imgs[:20]

    img_predictions = predictions_smv_task1(model, imgs)

    assert(len(img_predictions) == len(imgs))

    get_precision_recall_task1(img_predictions, desc)
    show_predictions(imgs, img_predictions)

def task1_hog_dense_nn():

    imgs, desc = get_train_data()

    pos_hog, _, neg_hog = get_pos_neg_hog(imgs, desc, 5)
    
    model = train_nn_task1(pos_hog, neg_hog)

    imgs, desc = get_validation_data()

    # TODO: remove
    imgs = imgs[:5]

    img_predictions = predictions_hog_w_dense_nn_task1(model, imgs)

    assert(len(img_predictions) == len(imgs))

    get_precision_recall_task1(img_predictions, desc)
    show_predictions(imgs, img_predictions)

def task1_cnn_validation():

    imgs, desc = get_train_data()

    pos_samples, _, neg_samples = get_traindata_cnn(imgs, desc, 5)
    model = train_cnn_task1(pos_samples, neg_samples)

    imgs, desc = get_validation_data()

    # TODO: remove
    imgs = imgs[:3]

    img_predictions = predictions_cnn_task1(model, imgs)

    assert(len(img_predictions) == len(imgs))

    get_precision_recall_task1(img_predictions, desc)
    show_predictions(imgs, img_predictions)

def task1_cnn_gridsearch():

    def _predictions_cnn_task1(model: Sequential, imgs, y_thresh, r_base):
    
        # resize image
        # equivalent to sliding windows of different sizes and shapes (rectangular / square)
        def _resized_img(img):
            
            #for resize_dim in [0.5, 0.75, 1, 1.25, 1.5]:
            #   for yx_ratio in [(2, 1), (1.7, 1), (1.5, 1), (1.25, 1), (1.1, 1), (1, 1), (1, 1.1), (1, 1.25)]:

            for resize_dim in [0.35, 0.5, 0.75, 1, 1.25, 1.5, 2]:
                for yx_ratio in [(2.2, 1), (1.9, 1), (1.5, 1), (1.25, 1), (1, 1)]:

                    yield resize_dim, yx_ratio, cv.resize(img, (int(resize_dim * img.shape[0] // yx_ratio[0]), int(resize_dim * img.shape[1] // yx_ratio[1])))

        # get coordinates for the (not necessarily square) window in the original image
        # corresponding to the square window in the reshaped image
        def _get_org_coords(resize_dim, yx_ratio, y1, x1):
            return int(y1 * yx_ratio[0] // resize_dim), int(x1 * yx_ratio[1] // resize_dim)
        
        # method responsible for filtering out 
        # some candidates that are most likely NOT faces
        def _get_y_matrix(img):

            # pad matrix with +1 to avoid out of bounds access further on
            # due to mul/div operations on some indexes

            y_mat = [[0 for _ in range(img.shape[1] + 1)] for _ in range(img.shape[0] + 1)]
            r_mat = [[0 for _ in range(img.shape[1] + 1)] for _ in range(img.shape[0] + 1)]
            
            for i in range(img.shape[0]):
                
                if img[i][0][0] < 80 and img[i][0][1] > 100 and img[i][0][2] > 150:
                    y_mat[i][0] += 1

                if img[i][0][0] > 180 and img[i][0][1] > 180 and img[i][0][2] > 180:
                    r_mat[i][0] += 1

            for j in range(img.shape[1]):
                    
                if img[0][j][0] < 80 and img[0][j][1] > 100 and img[0][j][2] > 150:
                    y_mat[0][j] += 1

                if img[0][j][0] > 180 and img[0][j][1] > 180 and img[0][j][2] > 180:
                    r_mat[0][j] += 1
            
            for i in range(1, img.shape[0]):
                for j in range(1, img.shape[1]):
                    
                    if img[i][j][0] < 80 and img[i][j][1] > 100 and img[i][j][2] > 150:
                        y_mat[i][j] += 1

                    if img[i][j][0] > 180 and img[i][j][1] > 180 and img[i][j][2] > 180:
                        r_mat[i][j] += 1

                    y_mat[i][j] += y_mat[i - 1][j] + y_mat[i][j - 1] - y_mat[i - 1][j - 1]
                    r_mat[i][j] += r_mat[i - 1][j] + r_mat[i][j - 1] - r_mat[i - 1][j - 1]
            
            # assign values to padding cells

            for i in range(img.shape[0]):
                
                y_mat[i][img.shape[1]] = y_mat[i][img.shape[1] - 1]
                r_mat[i][img.shape[1]] = r_mat[i][img.shape[1] - 1]

            for j in range(img.shape[1]):
                
                y_mat[img.shape[0]][j] = y_mat[img.shape[0] - 1][j]
                r_mat[img.shape[0]][j] = r_mat[img.shape[0] - 1][j]

            y_mat[img.shape[0]][img.shape[1]] = y_mat[img.shape[0] - 1][img.shape[1] - 1]
            r_mat[img.shape[0]][img.shape[1]] = r_mat[img.shape[0] - 1][img.shape[1] - 1]

            return y_mat, r_mat

        def _get_y(x1, y1, x2, y2, y_mat, r_mat, use_eyes=True):

            def _logpow(x, exp):

                if exp <= 16:
                    return x ** exp

                if exp & 1 == 0:
                    return _logpow(x * x, exp // 2)

                return x * _logpow(x * x, exp // 2)

            val = y_mat[y2][x2] + y_mat[y1][x1] - y_mat[y2][x1] - y_mat[y1][x2]
            
            if use_eyes:

                R_BASE = r_base
                val *= _logpow(R_BASE, r_mat[y2][x2] + r_mat[y1][x1] - r_mat[y2][x1] - r_mat[y1][x2])

            return val / ((x2 - x1) * (y2 - y1))

        scores = []

        _cnt = 0
        for initial_img in imgs:

            print(f"prediction for idx {_cnt}")
            _cnt += 1

            scores.append([])

            Y_THRESH = y_thresh

            backup = None
            backup_y = -1

            y_mat, r_mat = _get_y_matrix(initial_img)

            for resize_dim, yx_ratio, img in _resized_img(initial_img):
                
                # I have the resized images, 
                # all chunks will be squares but they will cover the rectangular cases anyways
                # due to yx_ratio used

                ymax, xmax, _ = img.shape
                if ymax < FACE_LEN or xmax < FACE_LEN:
                    continue

                # filtering images that do not surpass yellow threeshold
                # and prepare the others for batch prediction

                for x1 in range(0, xmax + 1 - FACE_LEN, CELL_LEN):
                    for y1 in range(0, ymax + 1 - FACE_LEN, CELL_LEN):

                        img_preprocessed = img[y1: y1 + FACE_LEN, x1: x1 + FACE_LEN]

                        y1_org, x1_org = _get_org_coords(resize_dim, yx_ratio, x1, y1)
                        y2_org, x2_org = _get_org_coords(resize_dim, yx_ratio, x1 + FACE_LEN, y1 + FACE_LEN)

                        assert(y2_org <= initial_img.shape[0])
                        assert(x2_org <= initial_img.shape[1])

                        y = _get_y(x1_org, y1_org, x2_org, y2_org, y_mat, r_mat, True)

                        if y > Y_THRESH:
                            scores[-1].append([x1_org, y1_org, x2_org, y2_org, img_preprocessed, y])

                            # from now on, no need for backup
                            backup_y = 100000000
                            backup = None

                        elif y > backup_y: 

                            backup_y = y
                            backup = [[x1_org, y1_org, x2_org, y2_org, img_preprocessed, y]]

            if len(scores[-1]) > 0:

                print(f"idx {len(scores) - 1}, to predict cnt {len(scores[-1])}")

                batch = [s_[4] for s_ in scores[-1]]
                predicted = model.predict(np.array(batch))

                for pr_idx in range(len(predicted)):

                    pr = predicted[pr_idx]

                    assert(pr[0] + pr[1] > 0.99)

                    if pr[1] < pr[0]:
                        pr = -1
                    else:
                        pr = pr[1]

                    scores[-1][pr_idx][4] = pr

            elif len(scores[-1]) == 0:

                # at least, best worst case
                # this approach helps in the few cases when 
                # in the image even the "main" face does not surpass the y threshold
                # but at the same time we do not want to lower the y threshold

                pr = model(tf.convert_to_tensor([backup[0][4]]))[0]

                assert(pr[0] + pr[1] > 0.99)

                if pr[1] < pr[0]:
                    pr = -1
                else:
                    pr = pr[1]

                backup[0][4] = pr
                scores[-1] = backup

            # NOTE: the sliding window was a square, but it was applied on 
            #       a wide rande of resized versions of the initial image
            #       to cover as much cases as possible
            #       the non maximum suppression will be applied on all those 
            #       different-sized chunks

            scores[-1] = non_maximum_suppression(scores[-1])
            assert(scores[-1] is scores[len(scores) - 1])

        return scores

    def _task1_cnn(imgs, desc, vimgs, vdesc, h):

        pos_samples, _, neg_samples = get_traindata_cnn(imgs, desc, 5)
        model = train_cnn_task1(pos_samples, neg_samples)

        imgs, desc = vimgs, vdesc

        # TODO: remove
        #imgs = imgs[:10] + imgs[50: 60] + imgs[100: 110] + imgs[150: 160]
        #desc = desc[:10] + desc[50: 60] + desc[100: 110] + desc[150: 160]

        img_predictions = _predictions_cnn_task1(model, imgs, **h)
        return get_precision_recall_task1(img_predictions, desc)

    hyperparam_val ={
                        "y_thresh": [0.6],
                        "r_base": [1.001]
                    }
    hyperparam_names = [name for name in hyperparam_val.keys()]

    def _get_hyperparam_seq(params):

        if len(params) == 0:
            yield {}
        
        else:

            for val in hyperparam_val[params[0]]:
                for seq in _get_hyperparam_seq(params[1:]):

                    to_yield = {params[0]: val} 
                    to_yield.update(seq.copy())

                    yield to_yield

    imgs, desc = get_train_data()
    vimgs, vdesc = get_validation_data()

    for hyperparams in _get_hyperparam_seq(hyperparam_names):

        try:
            p, r = _task1_cnn(deepcopy(imgs), deepcopy(desc), deepcopy(vimgs), deepcopy(vdesc), hyperparams)

            with open("cnn_gridsearch.txt", "a") as f:
                f.write(f"precision {p}, recall {r}, params: {hyperparams}\n")

        except Exception as err:

            with open("cnn_gridsearch.txt", "a") as f:
                f.write(f"error {err.args}, params: {hyperparams}\n")

def task1_cnn():

    imgs, desc = get_train_data()

    pos_samples, _, neg_samples = get_traindata_cnn(imgs, desc, 5)
    model = train_cnn_task1(pos_samples, neg_samples)

    imgs, img_names = get_test_data(f"{VALIDATION_DATA_PATH}/simpsons_validare/")

    # TODO: remove
    # imgs = imgs[:3]

    img_predictions = predictions_cnn_task1(model, imgs)

    assert(len(img_predictions) == len(imgs))
    assert(len(imgs) == len(img_names))

    save_predictions_task1(img_predictions, img_names)

# task 2 specific
# TODO

if __name__ == "__main__":
    
    #task1_cnn_validation()
    #task1_cnn_gridsearch()

    #get_test_data(f"{VALIDATION_DATA_PATH}/simpsons_validare/")
    task1_cnn()