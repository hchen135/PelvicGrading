import pandas as pd
from causalnex.evaluation import roc_auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from copy import deepcopy
from collections import defaultdict 
from itertools import chain


def clean_result(RESULT_dict,reference_GT_dict,use_counting=True,total_counting=25):
    out = {}
    for key in reference_GT_dict:
        preds = {"21":0,"22":0,"31":0,"32":0,"9":0}

        if key in RESULT_dict:
            for bbox in RESULT_dict[key]["bboxes"]:
                label = bbox["label"]
                if use_counting:
                    valid_counting = bbox["valid_counting"]
                    prob = valid_counting/total_counting
                else:
                    prob = bbox["confidence"]
                if preds[str(label)] < prob:
                    preds[str(label)] = prob

        out[key] = preds

    return out




def GT_label_to_dict(GT_labels_df,age_threshold=60):
    out = {}

    subjects = [i.split('\\')[-1] for i in GT_labels_df["filepath"]]
    tile = GT_labels_df["after arbitration"]
    age = GT_labels_df["Age"]


    tile_dict = {"A":1,"B":2,"C":3}
    for i in range(len(subjects)):
        name_i = subjects[i]
        tile_i = tile_dict[tile[i]]
        if age[i] is None:
            age_i = 0 
        else:
            age_i = int(age[i] >= age_threshold)
        r_i = int(tile_i >= 2)
        t_i = int(tile_i == 3)

        if 'test' in name_i:
            name_i = 'PAM test '+name_i[7:]
        elif 'pro' in name_i:
            name_i = 'PAM pro '+name_i[6:]
        elif 'BU' in name_i:
            name_i = name_i
        else:
            name_i = 'PAM '+name_i[3:]

        out[name_i] = {'r':r_i,'t':t_i,'age':age_i,'tile':tile_i}
    return out


'''
### old version
def r_t_tile_form(data_dict_ori,GT_phase=True,threshold=0):
    data_dict = deepcopy(data_dict_ori)
    for key in data_dict:
        r = 0
        t = 0
        tile = 1
        if data_dict[key]["31"] > threshold:
            r = 1
            data_dict[key]["31"] = 1
        else:
            data_dict[key]["31"] = 0

        if data_dict[key]["32"] > threshold:
            r = 1
            t = 1
            data_dict[key]["32"] = 1
        else:
            data_dict[key]["32"] = 0

        if data_dict[key]["21"] > threshold:
            r = 1
            data_dict[key]["21"] = 1
        else:
            data_dict[key]["21"] = 0

        if data_dict[key]["22"] > threshold:
            r = 1
            t = 1
            data_dict[key]["22"] = 1
        else:
            data_dict[key]["22"] = 0

        if data_dict[key]["9"] > threshold:
            r = 1
            t = 1
            data_dict[key]["9"] = 1
        else:
            data_dict[key]["9"] = 0

        data_dict[key]["r"] = r
        data_dict[key]["t"] = t
        if r == 1:
            tile = 2
        if t == 1:
            tile = 3
        data_dict[key]["tile"] = tile 

    return data_dict
'''

def r_t_tile_form(data_dict_ori,GT_label_dict):
    data_dict = {}
    _count = 0
    for i in data_dict_ori:
        if i in GT_label_dict:
            _count +=1
            data_dict[i] = data_dict_ori[i]
            data_dict[i]['r'] = GT_label_dict[i]['r']
            data_dict[i]['t'] = GT_label_dict[i]['t']
            data_dict[i]['tile'] = GT_label_dict[i]['tile']
            data_dict[i]['age'] = GT_label_dict[i]['age']

    print(_count)
    return data_dict

def threshold_set(data_dict_ori,threshold,label_key=['21','22','31','32','9']):
    data_dict = deepcopy(data_dict_ori)
    for i in data_dict:
        for key in label_key:
            data_dict[i][key] = int(data_dict[i][key] >= threshold)
    return data_dict

def train_val_subject_gene(subjects,num_fold=5):
    train_subjects = []
    val_subjects = []

    i = 0
    tmp_val = []
    while i < len(subjects):
        tmp_val.append(subjects[i])
        if i % (len(subjects)//num_fold+1) == len(subjects)//num_fold:
            val_subjects.append(tmp_val)
            train_subjects.append([i for i in subjects if i not in tmp_val])
            print(len(tmp_val))
            tmp_val = []

        i+=1

    if len(tmp_val) > 0:
        val_subjects.append(tmp_val)
        train_subjects.append([i for i in subjects if i not in tmp_val])
        print(len(tmp_val))

    return train_subjects,val_subjects

def df_gene(data_dict,subject_list,column_names=['name','age','a21','a22','a31','a32','a9','r','t','tile']):

    out = []
    for subject in subject_list:
        tmp = []
        tmp.append(subject)
        tmp.append(data_dict[subject]['age'])
        for i in range(2,len(column_names)):
            if column_names[i].startswith('a'):
                # print(data_dict[subject])
                tmp.append(data_dict[subject][column_names[i][1:]])
            else:
                tmp.append(data_dict[subject][column_names[i]])
        out.append(tmp)

    return pd.DataFrame(out,columns=column_names)

def get_rt_options(t,r):
    if r == 0 and t == 0:
        return ([0,1],[1,1])
    if r == 1 and t == 0:
        return ([0,0],[1,1])
    if r == 1 and t == 1:
        return ([0,1],[0,0])

def predict_rt(model, df, result_dict):
    predictions_r = model.predict(df, "r")
    predicted_probabilities = model.predict_probability(df, "r")
    
    #print("r",predictions)
    counter = 0
    j = 0
    predict = []
    expert = []
    for i in predictions_r['r_prediction']:
        if i==df["r"][j]:
            counter +=1
        
        predict.append(int(i))
        expert.append(int(df["r"][j]))
        j +=1
        
    roc, auc = roc_auc(model, df, "r")
    
    #print(classification_report(model, df, "r"))
    print("Acc rotational instability",round(counter/len(df),2),"Kappa",round(cohen_kappa_score(predict, expert, weights='linear'),2),"AUC:",round(auc,2))

    result_dict["ROT_acc"].append(counter/len(df))
    result_dict["ROT_kappa"].append(cohen_kappa_score(predict, expert))
    result_dict["ROT_auc"].append(auc)

    predictions_t = model.predict(df, "t")
    predicted_probabilities = model.predict_probability(df, "t")
    
    #print("t",predictions)
    counter = 0
    j = 0
    predict = []
    expert = []
    for i in predictions_t['t_prediction']:
        if i==df["t"][j]:
            counter +=1
        
        predict.append(int(i))
        expert.append(int(df["t"][j]))
        j +=1
        
        
    roc, auc = roc_auc(model, df, "t")
    #print(classification_report(model, df, "t"))
    print("Acc translational instability",round(counter/len(df),2),"Kappa",round(cohen_kappa_score(predict, expert, weights='linear'),2),"AUC:",round(auc,2))

    result_dict["TRA_acc"].append(counter/len(df))
    result_dict["TRA_kappa"].append(cohen_kappa_score(predict, expert))
    result_dict["TRA_auc"].append(auc)


def refine_predictions(df, df_low_conf, ie, THRESH=0.61, iters=1, no_age=False, logging=False):
    dataset_refined = pd.DataFrame(columns=['age','a21','a22','a31','a32','a9','name','t','r'], dtype='int')
    col_names = ['age','a21','a22','a31','a32','a9','name','t','r']
    
    for k,row in df.iterrows():
        r =  row['r']
        t =  row['t']
        name = row['name']
    
        del row['name']
        del row['t']
        del row['r']
    
        confirmed_dict = defaultdict(int)
        for key in row.to_dict().keys():
            if key =='age' and no_age:
                continue
            if row.to_dict()[key]==1:
                confirmed_dict[key]=1
            if key =='age':
                confirmed_dict[key]=row.to_dict()[key]
        new_detections, missed = recompute_posteriors(ie, confirmed_dict,THRESH, False)
        if logging:
            print(name)
            print("Dict", confirmed_dict)
            print("Missed?", missed)
            print("ND",new_detections)
            print(df_low_conf.loc[df_low_conf['name'] == name])
            
        low_conf_contained = check_if_low_conf_contains(missed, df_low_conf.loc[df_low_conf['name'] == name])
        new_detections = dict(chain.from_iterable(d.items() for d in (confirmed_dict, low_conf_contained)))
        
        if logging:
            print("Low found", low_conf_contained)
            print("New + old", confirmed_dict, low_conf_contained)
            print("Updated query",new_detections)
            print()
        for i in range(1, iters):
            new_detections, missed = recompute_posteriors(ie, new_detections,THRESH,False)
            if logging:
                print(i)
                print("Missed?", missed)
                print("ND",new_detections)
                print(name)
                print(df_low_conf.loc[df_low_conf['name'] == name])
                
            low_conf_contained = check_if_low_conf_contains(missed, df_low_conf.loc[df_low_conf['name'] == name])
            new_detections = dict(chain.from_iterable(d.items() for d in (confirmed_dict, low_conf_contained)))
            
            if logging:
                print("Low found", low_conf_contained)
                print("New + old", confirmed_dict, low_conf_contained)
                print("Updated query",new_detections)
                print()
        
        new_dict_full = defaultdict(int)
        for i in range(len(col_names)):
            if col_names[i] in new_detections.keys():
                new_dict_full[col_names[i]]=new_detections[col_names[i]]
            elif col_names[i] == 'r':
                new_dict_full[col_names[i]]=r
            elif col_names[i] == 't':
                new_dict_full[col_names[i]]=t
            elif col_names[i] == 'name':
                new_dict_full[col_names[i]]=name
            else:
                new_dict_full[col_names[i]]=0
            
        new_line = pd.DataFrame({"age":new_dict_full["age"], 
                             "a21":new_dict_full["a21"],'a22':new_dict_full["a22"],
                             'a31':new_dict_full["a31"],'a32':new_dict_full["a32"],
                             'a9':new_dict_full["a9"],'name':new_dict_full["name"],
                             't':new_dict_full["t"],'r':new_dict_full["r"]}, index=[0]) 
        dataset_refined = dataset_refined.append(new_line) 

    dataset_refined=dataset_refined.reset_index()
    return dataset_refined

def recompute_posteriors(ie, dict_query,THRESH=0.61, log = True):
    t_dict_query = dict(dict_query)
    try:
        del t_dict_query["index"]
    except:
        pass
    try:
        del t_dict_query["test"]
    except:
        pass
    
    marginals_after_observations = ie.query(t_dict_query)
    if marginals_after_observations['r'][0]>marginals_after_observations['r'][1]:
        r = 0
    else:
        r = 1
        
    if marginals_after_observations['t'][0]>marginals_after_observations['t'][1]:
        t = 0
    else:
        t = 1
        
    if log:
        print(marginals_after_observations)
        print("Given ",get_text_query(dict_query),
          " current grade r:",str(r),
          " t:",str(t))
    rt_options = get_rt_options(t,r)
    
    clean_dict = defaultdict(dict)
    possibly_missed = defaultdict(int)
    
    for key in marginals_after_observations:
        if 1.>marginals_after_observations[key][1]>THRESH and key != 'r' and key != 't':
            possibly_missed[key] = 1
    if len(possibly_missed)>0 and log:
        print("Possibly missed", get_text_query(possibly_missed))
    
    #updated_dict = dict(chain.from_iterable(d.items() for d in (possibly_missed, dict_query)))
    updated_dict = dict(dict_query)
    possibly_missed2=defaultdict(int)
    # if r and t will change
    for elem in rt_options:
        new_t,new_r = elem
        t_dict_query["r"]=new_r
        t_dict_query["t"]=new_t
        t_marginals_after_observations = ie.query(t_dict_query)
        for key in t_marginals_after_observations:
            clean_dict[key]={0:abs(t_marginals_after_observations[key][0]-marginals_after_observations[key][0]),
                             1:abs(t_marginals_after_observations[key][1]-marginals_after_observations[key][1])}
            
            #if 1.>t_marginals_after_observations[key][0]>THRESH and 1>marginals_after_observations[key][1]>THRESH:
            #    possibly_missed[key] = 0
            if 1.>t_marginals_after_observations[key][1]>THRESH and 1>marginals_after_observations[key][0]>THRESH:
                possibly_missed2[key] = 1

        if len(possibly_missed2)>0 and log:
            print("Current grade r:" + str(r)+" t:"+str(t)+" ---> new r:"+str(new_r)+" new t:"+str(new_t))
            print()
            print("In case this fracture/s is missed, grading could change:", get_text_query(possibly_missed2), possibly_missed2)
            
        possibly_missed = dict(chain.from_iterable(d.items() for d in (possibly_missed, possibly_missed2)))
        possibly_missed2 = defaultdict(int)
    #print("PM",possibly_missed) # check whatever these are present in the low-confidence dictionary
    return updated_dict, possibly_missed

def check_if_low_conf_contains(missed_dict, low_conf_dict):
    checked_dict = defaultdict(int)
    for key in missed_dict.keys():
        #dataset_m
        #print(low_conf_dict['name'])
        if low_conf_dict.iloc[0][key] == 1:
            checked_dict[key] = 1
    return checked_dict

def fracture_avg_status(frac_prob,frac_y,frac_stat_csv,RESULT_dict,GT_dict,considered_fractures = ['a21','a22','a31','a32','a9']):
    stat = frac_stat_csv.values 
    column_names = frac_stat_csv.columns.values
    for i in range(len(column_names)):
        if column_names[i] == 'name':
            name_column = int(i)
            break
    # print(column_names)
    for i in range(len(stat)):
        subject_name = stat[i][name_column]
        # print(subject_name)
        for j in range(2,7):
            class_name = column_names[j]
            if class_name not in considered_fractures:
                continue
            class_name = class_name[1:] 
            stat_value = stat[i][j]
            if stat_value > 0:
                frac_prob.append(RESULT_dict[subject_name][class_name])
            else:
                frac_prob.append(0)
            frac_y.append(GT_dict[subject_name][class_name])










