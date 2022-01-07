import numpy as np
import cv2 as cv
from math import ceil
from random import randint
from copy import deepcopy
from os import listdir

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import SGD

TEST_DATA_PATH = "../testare/simpsons_testare/"
PREDICTION_PATH_TASK1 = "../fisiere_solutie/Stanciu_Andrei_Calin_331/task1/"
PREDICTION_PATH_TASK2 = "../fisiere_solutie/Stanciu_Andrei_Calin_331/task2/"

# I ve included the train data and kept the training code inside this file
# for emergency situations
# If the models are loaded from the files without problems, this should not be needed
TRAIN_DATA_PATH = "./antrenare/"

MODEL_INPUT_DIM = 72
WINDOW_STRIDE = 8

IOU_NMS_THRESHOLD = 0.15

# custom keras stuff
# NOTE: for custom layers and resnet, examples and documentation source:
#       * https://d2l.ai/chapter_convolutional-modern/resnet.html
#       * https://www.tensorflow.org/tutorials/customization/custom_layers
#       * https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class ResidualId(Layer):

    def __init__(self, filters, kernel_size=(3, 3), strides=1):
        super(ResidualId, self).__init__()

        self.conv0 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides)
        self.n0 = BatchNormalization()
        self.a0 = Activation('relu')

        self.conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.n1 = BatchNormalization()

        self.a1 = Activation('relu')

    def call(self, input_tensor):
        
        _tmp = self.conv0(input_tensor)
        _tmp = self.n0(_tmp)
        _tmp = self.a0(_tmp)

        _tmp = self.conv1(_tmp)
        _tmp = self.n1(_tmp)

        _tmp += input_tensor

        output_tensor = self.a1(_tmp)
        return output_tensor

class ResidualConv(Layer):

    # additional convolution for skip path to be able to change output channel count

    def __init__(self, filters, kernel_size=(3, 3), strides=1):
        super(ResidualConv, self).__init__()

        self.conv0 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides)
        self.n0 = BatchNormalization()
        self.a0 = Activation('relu')

        self.conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.n1 = BatchNormalization()

        self.conv_input = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides)

        self.a1 = Activation('relu')

    def call(self, input_tensor):
        
        _tmp = self.conv0(input_tensor)
        _tmp = self.n0(_tmp)
        _tmp = self.a0(_tmp)

        _tmp = self.conv1(_tmp)
        _tmp = self.n1(_tmp)

        _tmp2 = self.conv_input(input_tensor)
        _tmp += _tmp2

        output_tensor = self.a1(_tmp)
        return output_tensor

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

def get_test_data(path):

    img_names = [n for n in listdir(path)]
    imgs = [cv.imread(f"{path}{img_n}") for img_n in img_names]

    return imgs, img_names

def show_predictions(imgs, predictions):

    for i in range(len(imgs)):
        
        img = imgs[i]
        for pred in predictions[i]:

            x1, y1, x2, y2 = pred[0], pred[1], pred[2], pred[3]

            cv.line(img, (x1, y1), (x2, y1), color=(255, 0, 0))
            cv.line(img, (x1, y1), (x1, y2), color=(255, 0, 0))
            cv.line(img, (x2, y1), (x2, y2), color=(255, 0, 0))
            cv.line(img, (x1, y2), (x2, y2), color=(255, 0, 0))

        cv.imshow(f"prediction idx {i}", img)
        cv.waitKey(0)

def save_predictions_task1(img_predictions, img_names):

    detections = []
    scores = []
    file_names = []

    for idx in range(len(img_predictions)):

        for p in img_predictions[idx]:

            detections.append([p[0], p[1], p[2], p[3]])
            scores.append(p[4])
            file_names.append(img_names[idx])

            # print(f"predicted with wsize {p[5]}, y {p[6]}")

    detections = np.array(detections)
    scores = np.array(scores)
    file_names = np.array(file_names)

    np.save(f"{PREDICTION_PATH_TASK1}detections_all_faces", detections)
    np.save(f"{PREDICTION_PATH_TASK1}scores_all_faces", scores)
    np.save(f"{PREDICTION_PATH_TASK1}file_names_all_faces", file_names)

def save_predictions_task2(img_predictions, img_filenames):

    detections = {"bart": [], "homer": [], "lisa": [], "marge": []}
    scores = {"bart": [], "homer": [], "lisa": [], "marge": []}
    file_names = {"bart": [], "homer": [], "lisa": [], "marge": []}

    for idx in range(len(img_predictions)):

        for p in img_predictions[idx]:

            name = p[5]

            # in current implementation, this should always be true at this point
            if name in ["bart", "marge", "homer", "lisa"]:

                detections[name].append([p[0], p[1], p[2], p[3]])
                scores[name].append(p[4])
                file_names[name].append(img_filenames[idx])

            else:
                print("warn: unexpected branch in save predictions for task 2")

    for name in ["bart", "homer", "lisa", "marge"]:

        detections[name] = np.array(detections[name])
        scores[name] = np.array(scores[name])
        file_names[name] = np.array(file_names[name])

        np.save(f"{PREDICTION_PATH_TASK2}detections_{name}", detections[name])
        np.save(f"{PREDICTION_PATH_TASK2}scores_{name}", scores[name])
        np.save(f"{PREDICTION_PATH_TASK2}file_names_{name}", file_names[name])

def load_model_(path):
    
    model = load_model(path, custom_objects={'ResidualId': ResidualId, 'ResidualConv': ResidualConv})
    return model
    
# training and feature extraction

def get_train_data(imgs, desc, neg_factor=1):

    def _preprocess(img_):

        img_ = cv.resize(img_, (MODEL_INPUT_DIM, MODEL_INPUT_DIM))
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

            pos_samples.append(cv.flip(face, 1))
            pos_names.append(name)

            pos_cnt += 2

    neg_cnt = ceil(pos_cnt * neg_factor)

    neg_samples = []

    def _get_coord4neg(img, check_against):
        
        while True:

            y_ = randint(0, img.shape[0] - MODEL_INPUT_DIM)
            x_ = randint(0, img.shape[1] - MODEL_INPUT_DIM)

            ok = True
            for j in range(len(check_against)):

                x1, y1 = check_against[j][0]
                x2, y2 = check_against[j][1]

                if (x1 <= x_ <= x2 and y1 <= y_ <= y2) or (x1 <= x_ + MODEL_INPUT_DIM <= x2 and y1 <= y_ + MODEL_INPUT_DIM <= y2):
                    ok = False
                    break

            if ok:
                break

        return x_, y_

    for i in range(len(imgs)):
        img = imgs[i]

        for _ in range(ceil(neg_factor)):
            
            x_, y_ = _get_coord4neg(img, desc[i])

            img_processed = _preprocess(img[y_: y_ + MODEL_INPUT_DIM, x_: x_ + MODEL_INPUT_DIM, :])
            neg_samples.append(img_processed)

    neg_samples = np.array(neg_samples)

    return pos_samples, pos_names, neg_samples[:neg_cnt]

def train_resnet_task1(pos_samples, neg_samples):

    train_data = np.concatenate((np.array(pos_samples), neg_samples))
    train_labels = np.array([1 for _ in range(len(pos_samples))] + 
                            [0 for _ in range(len(neg_samples))])

    train_labels = tf.one_hot(train_labels, depth=2)

    # ResNet18, or at least something that resembles it

    model = Sequential([

        Conv2D(64, input_shape=(MODEL_INPUT_DIM, MODEL_INPUT_DIM, 3), kernel_size=(7, 7), strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=3, strides=2, padding='same'),

        ResidualId(64, kernel_size=(3, 3)),
        ResidualId(64, kernel_size=(3, 3)),

        ResidualConv(128, kernel_size=(3, 3), strides=2),
        ResidualId(128, kernel_size=(3, 3)),

        ResidualConv(256, kernel_size=(3, 3), strides=2),
        ResidualId(256, kernel_size=(3, 3)),

        ResidualConv(512, kernel_size=(3, 3), strides=2),
        ResidualId(512, kernel_size=(3, 3)),

        GlobalAvgPool2D(),
        Dense(2),
        Activation('softmax')
    ])

    model.compile(loss='binary_crossentropy', optimizer=SGD(0.0001, momentum=0.9), metrics=['accuracy'])
    model.fit(x=train_data, y=train_labels, epochs=5, batch_size=64)

    model.save(f"model_task1_{randint(0, 1000)}")

    return model

def train_resnet_task2(pos_samples, pos_names, neg_samples):

    train_data = np.concatenate((np.array(pos_samples), neg_samples))

    BART = 0
    HOMER = 1
    LISA = 2
    MARGE = 3
    UNKNOWN = 4
    NEGATIVE = 5
    
    train_labels = []
    for s in pos_names:

        if s == "bart":
            train_labels.append(BART)

        if s == "homer":
            train_labels.append(HOMER)

        if s == "lisa":
            train_labels.append(LISA)

        if s == "marge":
            train_labels.append(MARGE)
        
        # change NEGATIVE to UNKNOWN to separate unknown faces from "negative faces"
        if s == "unknown":
            train_labels.append(NEGATIVE)

    train_labels += [NEGATIVE for _ in range(len(neg_samples))]

    train_labels = tensorflow.keras.utils.to_categorical(train_labels, 6)

    # ResNet18, or at least something that resembles it

    model = Sequential([

        Conv2D(64, input_shape=(MODEL_INPUT_DIM, MODEL_INPUT_DIM, 3), kernel_size=(7, 7), strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=3, strides=2, padding='same'),

        ResidualId(64, kernel_size=(3, 3)),
        ResidualId(64, kernel_size=(3, 3)),

        ResidualConv(128, kernel_size=(3, 3), strides=2),
        ResidualId(128, kernel_size=(3, 3)),

        ResidualConv(256, kernel_size=(3, 3), strides=2),
        ResidualId(256, kernel_size=(3, 3)),

        ResidualConv(512, kernel_size=(3, 3), strides=2),
        ResidualId(512, kernel_size=(3, 3)),

        GlobalAvgPool2D(),
        Dense(6),
        Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=SGD(0.0001, momentum=0.9), metrics=['accuracy'])
    model.fit(x=train_data, y=train_labels, epochs=7, batch_size=64)

    model.save(f"model_task2_{randint(0, 1000)}")

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
            print('should not reach this: negative score inside NMS')
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

def predictions_task1(model: Sequential, imgs):

    # for yellow thresholds, model appears pretty sensitive
    # (eg., for low 0.2 and upper 0.75, the ap goes down significantly)
    
    Y_LOW_THRESH = 0.3         # avoid chunks that are not "body parts"
    Y_UPPER_THRESH = 0.7       # avoid full-yellow chunks 
    
    # calculates DP matrix for yellow pixel count
    def _get_y_matrix(img):

        # pad matrix with +1 to avoid out of bounds access further on
        # due to mul/div operations on some indexes

        y_mat = [[0 for _ in range(img.shape[1] + 1)] for _ in range(img.shape[0] + 1)]
        
        for i in range(img.shape[0]):
            
            if img[i][0][0] < 80 and img[i][0][1] > 130 and img[i][0][2] > 150:
                y_mat[i][0] += 1

        for j in range(img.shape[1]):
                
            if img[0][j][0] < 80 and img[0][j][1] > 130 and img[0][j][2] > 150:
                y_mat[0][j] += 1

        for i in range(1, img.shape[0]):
            for j in range(1, img.shape[1]):
                
                if img[i][j][0] < 80 and img[i][j][1] > 130 and img[i][j][2] > 150:
                    y_mat[i][j] += 1

                y_mat[i][j] += y_mat[i - 1][j] + y_mat[i][j - 1] - y_mat[i - 1][j - 1]
        
        # assign values to padding cells

        for i in range(img.shape[0]):
            y_mat[i][img.shape[1]] = y_mat[i][img.shape[1] - 1]

        for j in range(img.shape[1]):
            y_mat[img.shape[0]][j] = y_mat[img.shape[0] - 1][j]

        y_mat[img.shape[0]][img.shape[1]] = y_mat[img.shape[0] - 1][img.shape[1] - 1]

        return y_mat

    def _get_y(x1, y1, x2, y2, y_mat):

        val = y_mat[y2][x2] + y_mat[y1][x1] - y_mat[y2][x1] - y_mat[y1][x2]

        return val / ((x2 - x1) * (y2 - y1))

    def wsize_range():

        for wsize_y_factor in [1, 1.4]:
            for wsize_x in range(20, 110, 10):

                yield wsize_x, int(wsize_x * wsize_y_factor)

    # format of scores:
    #   scores[image index] = [list of [x1, y1, x2, y2, image / score on it, auxiliary field, auxiliary field]]
    # * I avoided the usage of an object with equivalent properties
    # in the hope that the code will run faster
    # * The auxiliary fields should not change the corectness, but were used for debugging / stats

    scores = []

    _cnt = 0
    for img in imgs:

        print(f"[TASK 1] prediction for image with idx {_cnt}")
        _cnt += 1

        scores.append([])

        xmax, ymax = img.shape[1], img.shape[0]

        y_mat = _get_y_matrix(img)

        # every window size with step 1 also works
        # but it is way slower, and does not show improvements
        for wsize_x, wsize_y in wsize_range():
        
            # filtering images that do not surpass yellow threeshold
            # and prepare the others for batch prediction

            for x1 in range(0, xmax + 1 - wsize_x, WINDOW_STRIDE):
                for y1 in range(0, ymax + 1 - wsize_y, WINDOW_STRIDE):

                    img_preprocessed = cv.resize(img[y1: y1 + wsize_y, x1: x1 + wsize_x], (MODEL_INPUT_DIM, MODEL_INPUT_DIM))

                    y = _get_y(x1, y1, x1 + wsize_x, y1 + wsize_y, y_mat)

                    if Y_UPPER_THRESH > y > Y_LOW_THRESH:
                        scores[-1].append([x1, y1, x1 + wsize_x, y1 + wsize_y, img_preprocessed, (wsize_x, wsize_y), y])

        if len(scores[-1]) > 0:
            
            batch = [s_[4] for s_ in scores[-1]]
            predicted = model.predict(np.array(batch))

            assert(len(predicted) == len(scores[-1]))

            _scores = []
        
            for pr_idx in range(len(predicted)):
                
                if predicted[pr_idx][1] > predicted[pr_idx][0]:
                    
                    scores[-1][pr_idx][4] = predicted[pr_idx][1]
                    _scores.append(deepcopy(scores[-1][pr_idx]))

        scores[-1] = non_maximum_suppression(_scores)

    return scores

def task1_run(load=True):

    if load is True:
        model = load_model_("model_task1")

    else:

        imgs, desc = get_train_data()

        pos_samples, _, neg_samples = get_train_data(imgs, desc, 4)
        model = train_resnet_task1(pos_samples, neg_samples)

    imgs, img_names = get_test_data(TEST_DATA_PATH)

    img_predictions = predictions_task1(model, imgs)

    assert(len(img_predictions) == len(imgs))

    save_predictions_task1(img_predictions, img_names)
    
# task 2 specific

def predictions_task2(model: Sequential, imgs):

    # for yellow thresholds, model appears pretty sensitive
    # (eg., for low 0.2 and upper 0.75, the ap goes down significantly)
    
    Y_LOW_THRESH = 0.3         # avoid chunks that are not "body parts"
    Y_UPPER_THRESH = 0.7       # avoid full-yellow chunks 
    
    # calculates DP matrix for yellow pixel count
    def _get_y_matrix(img):

        # pad matrix with +1 to avoid out of bounds access further on
        # due to mul/div operations on some indexes

        y_mat = [[0 for _ in range(img.shape[1] + 1)] for _ in range(img.shape[0] + 1)]
        
        for i in range(img.shape[0]):
            
            if img[i][0][0] < 80 and img[i][0][1] > 130 and img[i][0][2] > 150:
                y_mat[i][0] += 1

        for j in range(img.shape[1]):
                
            if img[0][j][0] < 80 and img[0][j][1] > 130 and img[0][j][2] > 150:
                y_mat[0][j] += 1

        for i in range(1, img.shape[0]):
            for j in range(1, img.shape[1]):
                
                if img[i][j][0] < 80 and img[i][j][1] > 130 and img[i][j][2] > 150:
                    y_mat[i][j] += 1

                y_mat[i][j] += y_mat[i - 1][j] + y_mat[i][j - 1] - y_mat[i - 1][j - 1]
        
        # assign values to padding cells

        for i in range(img.shape[0]):
            y_mat[i][img.shape[1]] = y_mat[i][img.shape[1] - 1]

        for j in range(img.shape[1]):
            y_mat[img.shape[0]][j] = y_mat[img.shape[0] - 1][j]

        y_mat[img.shape[0]][img.shape[1]] = y_mat[img.shape[0] - 1][img.shape[1] - 1]

        return y_mat

    def _get_y(x1, y1, x2, y2, y_mat):

        val = y_mat[y2][x2] + y_mat[y1][x1] - y_mat[y2][x1] - y_mat[y1][x2]

        return val / ((x2 - x1) * (y2 - y1))

    def wsize_range():

        for wsize_y_factor in [1, 1.4]:
            for wsize_x in range(20, 110, 10):

                yield wsize_x, int(wsize_x * wsize_y_factor)

    # format of scores:
    #   scores[image index] = [list of [x1, y1, x2, y2, image / score on it, auxiliary field, auxiliary field]]
    # * I avoided the usage of an object with equivalent properties
    # in the hope that the code will run faster
    # * The auxiliary fields should not change the corectness, but were used for debugging / stats

    scores = []

    _cnt = 0
    for img in imgs:

        print(f"[TASK 2] prediction for image with idx {_cnt}")
        _cnt += 1

        scores.append([])

        xmax, ymax = img.shape[1], img.shape[0]

        y_mat = _get_y_matrix(img)

        # every window size with step 1 also works
        # but it is way slower, and does not show improvements
        for wsize_x, wsize_y in wsize_range():
        
            # filtering images that do not surpass yellow threeshold
            # and prepare the others for batch prediction

            for x1 in range(0, xmax + 1 - wsize_x, WINDOW_STRIDE):
                for y1 in range(0, ymax + 1 - wsize_y, WINDOW_STRIDE):

                    img_preprocessed = cv.resize(img[y1: y1 + wsize_y, x1: x1 + wsize_x], (MODEL_INPUT_DIM, MODEL_INPUT_DIM))

                    y = _get_y(x1, y1, x1 + wsize_x, y1 + wsize_y, y_mat)

                    if Y_UPPER_THRESH > y > Y_LOW_THRESH:
                        scores[-1].append([x1, y1, x1 + wsize_x, y1 + wsize_y, img_preprocessed, (wsize_x, wsize_y), y])

        BART = 0
        HOMER = 1
        LISA = 2
        MARGE = 3
        UNKNOWN = 4
        NEGATIVE = 5

        convert_res = {BART: "bart", HOMER: "homer", LISA: "lisa", MARGE: "marge", UNKNOWN: "unknown", NEGATIVE: "negative"}

        if len(scores[-1]) > 0:
            
            batch = [s_[4] for s_ in scores[-1]]
            predicted = model.predict(np.array(batch))

            assert(len(predicted) == len(scores[-1]))

            _scores = []
        
            for pr_idx in range(len(predicted)):
                
                class_prediction_nr = np.argmax(predicted[pr_idx])
                class_prediction = convert_res[class_prediction_nr]
                
                if class_prediction in ["bart", "homer", "lisa", "marge"]:
                    
                    scores[-1][pr_idx][4] = predicted[pr_idx][class_prediction_nr]
                    scores[-1][pr_idx][5] = class_prediction
                    _scores.append(deepcopy(scores[-1][pr_idx]))

        scores[-1] = non_maximum_suppression(_scores)

    return scores

def task2_run(load=True):

    if load is True:
        model = load_model_("model_task2")

    else:

        imgs, desc = get_train_data()

        pos_samples, pos_names, neg_samples = get_train_data(imgs, desc, 4)
        model = train_resnet_task2(pos_samples, pos_names, neg_samples)

    imgs, img_names = get_test_data(TEST_DATA_PATH)

    img_predictions = predictions_task2(model, imgs)

    assert(len(img_predictions) == len(imgs))

    save_predictions_task2(img_predictions, img_names)

if __name__ == "__main__":

    try:

        task1_run(True)
        print("[TASK 1] done\n")

    except Exception as err:
        print(f"[TASK 1] error while running task 1: {err.args}")

    try:

        task2_run(True)
        print("[TASK 2] done\n")

    except Exception as err:
        print(f"[TASK 2] error while running task 1: {err.args}")

    print("[] predictions can be evaluated separately by running evalueaza_solutie.py")
