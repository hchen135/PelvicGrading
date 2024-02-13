from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import pandas
import pyshine as ps


import seaborn as sn
#import scikitplot as skplt

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import namedtuple

img_name_list = []
x1_list = []
x2_list = []
y1_list = []
y2_list = []
labels_list = []
conf_list =[]

sys.setrecursionlimit(40000)

DEFAULT_num_rois = 16
DEFAULT_network = 'restnet50'
DEFAULT_pam = '0'
arg_structure = namedtuple('arguments','test_path num_rois config_filename network pam')

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)

def contrast_scale_transform(img,factor):
    img = 128 + factor * (img - 128)
    return img 

def intensity_shift_transform(img,factor):
    img += factor
    return img

def intensity_scale_transform(img,factor):
    img *=factor
    return img

def five_fold_test(contrast_scale,intensity_scale,intensity_shift):
    print("*"*20+"start"+"*"*20)
    for fold_ind in range(5):
        options = arg_structure('./fold'+str(fold_ind)+'/test/images/tr',
                                DEFAULT_num_rois,
                                './fold'+str(fold_ind)+'/config_tr'+str(fold_ind)+'_fp.pickle',
                                DEFAULT_network,
                                DEFAULT_pam)


        # parser = OptionParser()

        # parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default="./fold3/test/images/tr")
        # parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
        #                   help="Number of ROIs per iteration. Higher means more memory use.", default=16)
        # parser.add_option("-c","--config_filename", dest="config_filename", help="Location to read the metadata related to the training (generated when training).",
        #                   default="./fold3/config_tr3_fp.pickle")
        # parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
        #                   default='resnet50')
        # parser.add_option("--pam", dest="pam", help="pam code", default='0')

        # (options, args) = parser.parse_args()

        if not options.test_path:  # if filename is not given
            parser.error('Error: path to test data must be specified. Pass --path to command line')

        config_output_filename = options.config_filename

        with open(config_output_filename, 'rb') as f_in:
            C = pickle.load(f_in)

        if C.network == 'resnet50':
            import keras_frcnn.resnet as nn
        elif C.network == 'vgg':
            import keras_frcnn.vgg as nn

        # turn off any data augmentation at test time
        C.use_horizontal_flips = False
        C.use_vertical_flips = False
        C.rot_90 = False

        img_path = options.test_path

        class_mapping = C.class_mapping

        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        class_mapping = {v: k for k, v in class_mapping.items()}
        print(class_mapping)
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
        C.num_rois = int(options.num_rois)

        if C.network == 'resnet50':
            num_features = 1024
        elif C.network == 'vgg':
            num_features = 512
        print(K.image_data_format())
        if K.image_data_format() == 'channels_first':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

        model_rpn = Model(img_input, rpn_layers)
        model_classifier_only = Model([feature_map_input, roi_input], classifier)

        model_classifier = Model([feature_map_input, roi_input], classifier)

        print('Loading weights from {}'.format(C.model_path))
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)

        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        all_imgs = []

        classes = {}

        bbox_threshold = 0.1

        visualise = True
        test_numbers = 0
        mis_wrong_pred = 0
        missed_count = {}
        pred = []
        label = []

        groupped_classes_dict = {11: 'Ring fx',
                                     12: 'Ring fx',
                                     21: 'Non-dias. Sac. fx',
                                     22: 'Dias. Sac. fx',
                                     31: 'AD SI',
                                     32: 'Parallel SI',
                                     9: 'PSD'}

        for idx, img_name in enumerate(sorted(os.listdir(img_path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)

            st = time.time()
            filepath = os.path.join(img_path, img_name)

            test_numbers += 1
            img = cv2.imread(filepath)


            #img = intensity_shift_transform(img,intensity_shift)
            #img = intensity_scale_transform(img,intensity_scale)
            #img = contrast_scale_transform(img,contrast_scale)
            X, ratio = format_img(img, C)
            X = X.astype(np.float32)

            if K.image_dim_ordering() == 'tf':
                X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)

            R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.2)
            if len(R) == 0:
                continue
            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0] // C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0] // C.num_rois:
                    # pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
                #print("F:",F)
                #print("ROIs:",ROIs)
                #print("P_cls:",P_cls)

                for ii in range(P_cls.shape[1]):

                    if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append(
                        [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []

            for key in bboxes:
                bbox = np.array(bboxes[key])
                try:
                    new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
                except:
                    continue
                #print("new_probs:",new_probs)
                for jk in range(new_boxes.shape[0]):
                    if key == "obj-fp":
                        continue
                    (x1, y1, x2, y2) = new_boxes[jk, :]
                    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                    # Here I created a list of 4 entries that contains the real coordinates
                    img_name_list.append(img_name)
                    x1_list.append(real_x1)
                    x2_list.append(real_x2)
                    y1_list.append(real_y1)
                    y2_list.append(real_y2)

                    #textLabel = '{}: {}'.format(groupped_classes_dict[int(key)], int(100 * new_probs[jk]))
                    textLabel = str(groupped_classes_dict[int(key)])
                    all_dets.append((key, 100 * new_probs[jk]))
                    labels_list.append(key)
                    #if key == 'obj-fp':
                    #    continue
                    cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                                  (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 1)


                    conf_list.append(100 * new_probs[jk])
                    (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, fontScale=0.1, thickness=1)
                    textOrg = (real_x1, real_y1 - 0)

                    size_rect = 1
                    #textLabel= textLabel.split(" ")[1]

                    #ps.putBText(img, textLabel, text_offset_x=real_x1-10, text_offset_y=real_y1-10, vspace=1, hspace=1,
                    #            font_scale=0.35, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, background_RGB=(228,225,222),text_RGB=(1,1,1))

                    #cv2.rectangle(img, (textOrg[0] - size_rect, textOrg[1] + baseLine - size_rect),
                    #              (textOrg[0] + retval[0] + size_rect, textOrg[1] - retval[1] - size_rect), (0, 0, 0), 1)
                    #cv2.rectangle(img, (textOrg[0] - size_rect, textOrg[1] + baseLine - size_rect),
                    #              (textOrg[0] + retval[0] + size_rect, textOrg[1] - retval[1] - size_rect), (255, 255, 255), -1)
                    #cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, fontScale=0.2, color=(0, 0, 0), thickness=1)

            print('Elapsed time = {}'.format(time.time() - st))
            print(all_dets)
            if len(all_dets):
                pred_class = (all_dets[0])[0]
                pred.append(pred_class)
                label.append(cls_name)
                if (pred_class != cls_name):
                    mis_wrong_pred += 1
            else:
                mis_wrong_pred += 1
                try:
                    if cls_name not in missed_count:
                        missed_count[cls_name] = 1
                    else:
                        missed_count[cls_name] += 1
                except:
                    pass

            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            scale_percent = 300  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite('./results_imgs/{}.png'.format(img_name), resized)

        ################# saving the results(bounding boxes) in a csv

        # df = pandas.DataFrame(data={"img_name": img_name_list, "x1": x1_list, "y1": y1_list, "x2": x2_list, "y2": y2_list,\
        #                             "label": labels_list})
        df_results = pandas.DataFrame(data={"img_name": img_name_list, "x1": x1_list, "y1": y1_list, "x2": x2_list, "y2": y2_list,\
                                    "label": labels_list, "confidence": conf_list})
        print(str(C.model_path))
        #df.to_csv("bb_coordinates"+str(C.model_path).split("/model")[1].split(".")[0]+".csv", sep=',', index=False)
        df_results.to_csv("2d_results/aug/fold"+str(fold_ind)+'_'+'IntensityShift'+str(intensity_shift)+'_IntensityScale'+str(intensity_scale)+'_ContrastScale'+str(contrast_scale)+".csv", sep=',', index=False)

        # ####### plotting into the cf matrix
        # class_label = ['11','21','22','31','32','9','bg']
        # array = confusion_matrix(label, pred, labels=class_label)

        # df_cm = pandas.DataFrame(array, index=[i for i in class_label],
        #                      columns=[i for i in class_label])

        # plt.figure(figsize=(40, 40))
        # sn.set(font_scale=1.4)  # for label size
        # #sn.heatmap(df_cm, annot=True)
        # skplt.metrics.plot_confusion_matrix(label, pred, normalize=True)
        # plt.savefig("conf_matr/conf_matrix"+str(C.model_path).split("/model")[1].split(".")[0]+".jpg")

        # #skplt.metrics.plot_confusion_matrix(label,pred, normalize=True)
        # #plt.show()
        # print("The missed count:", missed_count)
        # print('The missed and disclassified number', mis_wrong_pred)
        # print('The mean accuracy', (test_numbers - mis_wrong_pred) / test_numbers)

#for map result

#map_path = r'E:\Anna\Airplane-Detection-for-Satellites-master\mAP-master\mAP-master\input\\'
#fileout = map_path #+ "/detection-results/"
#df_results.to_csv(fileout+"wo"+options.pam+str(C.model_path).split("/model")[1].split(".")[0]+".csv", index=None, sep=',')



# contrast scale from 0.8 - 1.2
# intensity scale from 0.8 - 1.2
# intensity shift from -5 to 5


try:
    with open("finished.txt") as a:
        content = a.read().split('\n')[-2]
    contrast_scale_last = round(float(content.split("contrast scale: ")[-1].split(' finished')[0]),1)
    intensity_scale_last = round(float(content.split("intensity scale: ")[-1].split(',')[0]),1)
    intensity_shift_last = int(content.split('Intensity shift: ')[-1].split(',')[0])
except:
    contrast_scale_last = -100
    intensity_scale_last = -100
    intensity_shift_last = -100
 
for contrast_scale in np.arange(5)*0.1 + 0.8:
    if round(contrast_scale,2) < round(contrast_scale_last,2):
        continue
    for intensity_scale in np.arange(5)*0.1 + 0.8:
        if round(contrast_scale,2) == round(contrast_scale_last,2) and round(intensity_scale,2) < round(intensity_scale_last,2):
            continue
        for intensity_shift in np.arange(11)-5:
            if round(contrast_scale,2) == round(contrast_scale_last,2) and round(intensity_scale,2) == round(intensity_scale_last,2) and round(intensity_shift,2) <= round(intensity_shift_last,2):
                continue
            five_fold_test(contrast_scale,intensity_scale,intensity_shift)
            with open("finished.txt","a") as a:
                a.write("Intensity shift: "+str(int(intensity_shift)) + ", intensity scale: "+str(round(intensity_scale,2))+", contrast scale: "+str(round(contrast_scale,2))+" finished.\n")

# five_fold_test(1.0,1.0,0.0)
