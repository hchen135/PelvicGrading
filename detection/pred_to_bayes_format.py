import pandas as pd
from collections import defaultdict
import math


def tile_to_stabilities(tile):
    if tile == 'A':
        r, t = 0, 0
    elif tile == 'B':
        r, t = 1, 0
    elif tile == 'C':
        r, t = 1, 1
    return r,t

def get_test_set(fold_id):
    print("fold"+str(fold_id)+"/test/tr"+str(fold_id)+".csv")
    df_test = pd.read_csv("fold"+str(fold_id)+"/test/tr"+str(fold_id)+".csv")
    test_set = defaultdict(set)
    filepath = "E:\Anna\pelvic"
    for ind, row in df_test.iterrows():
        if len(row['img_name'].split(" ")[0].split("imgt")) == 2:
            pam_code=row['img_name'].split(" ")[0].split("imgt")[1]
            if "U" in pam_code:
                img_name = filepath + "\PAMB" + pam_code
            else:
                img_name = filepath + "\PAM" + pam_code
            test_set[img_name].add(row['label'])

    return test_set

def predictions_to_file(df, test_SET, CREATE_GT,CONF_THRESH):
    df_refined = pd.DataFrame(columns=['name', 'age', 'a21', 'a22', 'a31', 'a32', 'a9', 't', 'r'], dtype='int')
    df_refined_low_conf = pd.DataFrame(columns=['name', 'age', 'a21', 'a22', 'a31', 'a32', 'a9', 't', 'r'], dtype='int')

    groupped_classes_dict = {'a21': [5, 6, 17, 18], 'a22': [7, 8], 'a31': [9, 10], 'a32': [11, 12], 'a9': [24]}
    mapping_dict = {'a21': 21, 'a22': 22, 'a31': 31, 'a32': 32, 'a9': 9}

    # find those detections with conf that is higher than a CONF_THRESH
    filepath = "E:\Anna\pelvic"
    labels_dict_high_conf = defaultdict(set)
    for ind, row in df.iterrows():
        if len(row['img_name'].split(" ")[0].split("imgt")) == 2:
            img_name = filepath + "\PAM" + row['img_name'].split(" ")[0].split("imgt")[1]
            if float(row['confidence']) > CONF_THRESH:
                labels_dict_high_conf[img_name].add(row['label'])

    df = pd.read_csv("out.csv")
    print(test_SET.keys())

    # replace findings
    count = 0
    for ind, row in df.iterrows():
        img_name = row['filepath'].replace(" ", "")
        age = 0
        if row["age"] > 60:
            age = 1
        r, t = tile_to_stabilities(row["tile"])

        # saving the high conf into dict
        #if img_name in labels_dict_high_conf.keys():

        if img_name in test_SET.keys():
            count +=1
            df_refined = df_refined.append(
                    {'name': img_name, 'age': age, 'a21': 0, 'a22': 0, 'a31': 0, 'a32': 0, 'a9': 0,
                     't': t, 'r': r, 'test': int(1)}, ignore_index=True)
            if CREATE_GT:
                labels = row['labels'].replace("[", "").replace("]", "").split(",")
                labels = labels[0:len(labels)]
                labels = [int(i) for i in labels]
                unique_set = sorted(set(labels))
                for key, list_labels in groupped_classes_dict.items():
                    for j in unique_set:
                        for val in list_labels:
                            if int(val) == int(j):
                                df_refined.at[ind, str(key)] = 1
            else:
                #print(img_name)
                for k, v in mapping_dict.items():
                    for l in labels_dict_high_conf[img_name]:
                        if l == v:
                            df_refined.at[ind, k] = 1
        else:
            labels = row['labels'].replace("[", "").replace("]", "").split(",")
            labels = labels[0:len(labels)]
            labels = [int(i) for i in labels]
            unique_set = sorted(set(labels))
            df_refined = df_refined.append(
                    {'name': img_name, 'age': age, 'a21': 0, 'a22': 0, 'a31': 0, 'a32': 0, 'a9': 0,
                     't': t, 'r': r, 'test': int(0)}, ignore_index=True)

            for key, list_labels in groupped_classes_dict.items():
                for j in unique_set:
                    for val in list_labels:
                        if int(val) == int(j):
                            df_refined.at[ind, str(key)] = 1
    print("count",count)
    if CREATE_GT:
        df_refined.to_csv("bayes/f"+str(fold_id)+"/gt.csv", sep=',', index=False)
    else:
        df_refined.to_csv("bayes/f"+str(fold_id)+"/detected_" + str(round(CONF_THRESH * 0.01, 1)) + ".csv", sep=',', index=False)

for fold_id in range(0,5):
    CONF_THRESH = 51
    CREATE_GT = False
    test_s = get_test_set(fold_id)
    print(len(test_s))
    df = pd.read_csv("bb_tr" + str(fold_id) + "_fp.csv")
    predictions_to_file(df, test_s, CREATE_GT, CONF_THRESH)

    CONF_THRESH = 95
    CREATE_GT = False
    predictions_to_file(df, test_s, CREATE_GT, CONF_THRESH)

    CREATE_GT = True
    predictions_to_file(df, test_s, CREATE_GT, CONF_THRESH)