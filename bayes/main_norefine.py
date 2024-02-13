import pandas as pd 
import json
import os
import numpy as np
import argparse
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure
from causalnex.network import BayesianNetwork
from causalnex.discretiser import Discretiser
from causalnex.inference import InferenceEngine
from collections import defaultdict 
from causalnex.evaluation import classification_report
from util import clean_result,r_t_tile_form,df_gene,GT_label_to_dict,threshold_set,train_val_subject_gene,predict_rt,refine_predictions
np.random.seed(4)

parser = argparse.ArgumentParser()

parser.add_argument('--GT_FRAC_PATH',dest='GT_FRAC_PATH',type=str,default='/Users/hc/Documents/JHU/PJ/Mathias/AnnaProject/Pelvic/Journal_Extention/pelvic/GT/GT_dict.json')
parser.add_argument('--GT_LABEL_PATH',dest='GT_LABEL_PATH',type=str,default='/Users/hc/Documents/JHU/PJ/Mathias/AnnaProject/Pelvic/Journal_Extention/pelvic/GT/Automated Tile cases_Consensus_final consensus grades.csv')
parser.add_argument('--RESULT_PATH',dest='RESULT_PATH',type=str,default='/Users/hc/Documents/JHU/PJ/Mathias/AnnaProject/Pelvic/Journal_Extention/pelvic/final_results/out.json')
parser.add_argument('--high_threshold',dest='high_threshold',type=float,default=0.9)
parser.add_argument('--low_threshold',dest='low_threshold',type=float,default=0.7)
parser.add_argument('--threshold',dest='threshold',type=float,default=0.2)
parser.add_argument('--num_fold',dest='num_fold',type=int,default=5)
parser.add_argument('--anna',dest='anna',action='store_true')
parser.add_argument('--result_csv_name',dest='result_csv_name',type=str,default='all_results.csv')

args = parser.parse_args()


result_csv_name = args.result_csv_name


####################################
# DATA LOADING
####################################
with open(args.GT_FRAC_PATH) as a:
	GT_dict = json.load(a)

with open(args.RESULT_PATH) as a:
	RESULT_dict = json.load(a)

GT_labels_df = pd.read_csv(args.GT_LABEL_PATH)

####################################
# DATA PREP
####################################
# convert fracture detections into subject-level predictions 
if args.anna:
    RESULT_dict = clean_result(RESULT_dict,reference_GT_dict=GT_dict,use_counting=False)
else:
    RESULT_dict = clean_result(RESULT_dict,reference_GT_dict=GT_dict)

# First get r,t,tile info from GT_labels_df into dits
GT_label_dict = GT_label_to_dict(GT_labels_df)


# Second, insert r,t,tile information into the all dicts
# only include the keys that exist in the keys of GT_label_dict
GT_dict = r_t_tile_form(GT_dict,GT_label_dict)
RESULT_dict = r_t_tile_form(RESULT_dict,GT_label_dict)
subjects = sorted(list(GT_dict.keys()))

# Set 0/1 status for fractures in result dict with specific threhsold
high_threshold_dict = threshold_set(RESULT_dict,args.high_threshold)
low_threshold_dict = threshold_set(RESULT_dict,args.low_threshold)
# print('high_threshold_dict',high_threshold_dict)

####################################
# 5-fold cross validation
####################################
subjects = np.random.choice(subjects,len(subjects),replace=False)
train_subjects_all,val_subjects_all = train_val_subject_gene(subjects,args.num_fold)

# print('val_subjects:',val_subjects_all)

# print(GT_dict)

ALL_GT_result = {'ROT_acc':[],'ROT_auc':[],'ROT_kappa':[],'TRA_acc':[],'TRA_auc':[],'TRA_kappa':[]}
ALL_High_result = {'ROT_acc':[],'ROT_auc':[],'ROT_kappa':[],'TRA_acc':[],'TRA_auc':[],'TRA_kappa':[]}
ALL_Low_result = {'ROT_acc':[],'ROT_auc':[],'ROT_kappa':[],'TRA_acc':[],'TRA_auc':[],'TRA_kappa':[]}
for i in range(args.num_fold):
    train_subjects = train_subjects_all[i]
    test_subjects = val_subjects_all[i]
    train = df_gene(GT_dict,train_subjects)
    test = df_gene(GT_dict,test_subjects)
    test_high_conf = df_gene(high_threshold_dict,test_subjects)
    test_low_conf = df_gene(low_threshold_dict,test_subjects)


    # print('train')
    # print(train)
    # print('test')
    # print(test)
    # print('test_low_conf')
    # print(test_low_conf)
    # print('test_high_conf')
    # print(test_high_conf)

    ####################################
    # INIT BAYES MODEL
    ####################################
    sm_manual = StructureModel()
    sm_manual.add_edges_from(
            #[('a31', 'age'),('a32', 'age'),('a22', 'age'),('a21', 'age'),('a9', 'age'),
            # The first version of the graph (copied from the notebook)
            [('a31', 'r'), ('a32', 'r'),  ("t","r"),
                      ('a21', 'r'), ('a22', 'r'),('a9', 'r'), ('age', 'r'),
                      ('a31', 't'), ('a32', 't'), 
                      ('a21', 't'), ('a22', 't'),('a9', 't'), ('age', 't')
            ]#,
            # The second version of graph (according to the notebook graph)
            # [('a31', 'age'),('a32', 'age'),('a22', 'age'),('a21', 'age'),('a9', 'age'),('age','r'),('age','t'),('t','r'),
            #           ('a31', 'r'), ('a32', 'r'),
            #           ('a21', 'r'), ('a22', 'r'),('a9', 'r'), 
            #           ('a31', 't'), ('a32', 't'), 
            #           ('a21', 't'), ('a22', 't'),('a9', 't'), 
            # ]#,
            # The third version of graph (according to Anna's email)
            # [('age','t'),('age','r'), ('t','r'),
            #           ('a31', 'r'), ('a32', 'r'),
            #           ('a21', 'r'), ('a22', 'r'),('a9', 'r'), 
            #           ('a32', 't'), 
            #           ('a22', 't'),('a9', 't'), 
            # ]#,
            # #origin="expert",
        )

    bn = BayesianNetwork(sm_manual)
    # First, learn all the possible states of the nodes using the whole dataset
    bn.fit_node_states(train.drop(columns=["name"]))
    # Fit CPDs using the training dataset
    bn.fit_cpds(train.drop(columns=["name"]), method="BayesianEstimator", bayes_prior="K2")
    # bn.fit_cpds(train.drop(columns=["name"]), method="MaximumLikelihoodEstimator")

    ie = InferenceEngine(bn)


    test=test.drop(columns=['tile','name'])
    test_high_conf=test_high_conf.drop(columns=['tile','name'])
    test_low_conf=test_low_conf.drop(columns=['tile','name'])


    
    print("Ground Truth:")
    predict_rt(bn, test, ALL_GT_result)
    print()

    print("Predictions on High-confidence dictionary:")
    predict_rt(bn, test_high_conf, ALL_High_result)
    print()


    print("Predictions on Low-confidence dictionary:")
    predict_rt(bn, test_low_conf, ALL_Low_result)
    print()

    #print("Take ONE iteration and refine predictions w help of low-confidence dictionary:")
    #refined = refine_predictions(test_high_conf, test_low_conf, ie, args.threshold, 1)
    #predict_rt(bn, refined, ALL_Refine_result)
    #print()


####################################
# Save results
####################################

# Aggregate allcross validation results
for key in ALL_GT_result:
    ALL_GT_result[key] = round(np.average(ALL_GT_result[key]),3)
    ALL_High_result[key] = round(np.average(ALL_High_result[key]),3)
    ALL_Low_result[key] = round(np.average(ALL_Low_result[key]),3)
    #ALL_Refine_result[key] = round(np.average(ALL_Refine_result[key]),3)

all_data = [args.high_threshold,args.low_threshold,args.threshold]
all_data.extend([ALL_GT_result['ROT_acc'],ALL_High_result['ROT_acc'],ALL_Low_result['ROT_acc'],ALL_GT_result['TRA_acc'],ALL_High_result['TRA_acc'],ALL_Low_result['TRA_acc']])
all_data.extend([ALL_GT_result['ROT_kappa'],ALL_High_result['ROT_kappa'],ALL_Low_result['ROT_kappa'],ALL_GT_result['TRA_kappa'],ALL_High_result['TRA_kappa'],ALL_Low_result['TRA_kappa']])
all_data.extend([ALL_GT_result['ROT_auc'],ALL_High_result['ROT_auc'],ALL_Low_result['ROT_auc'],ALL_GT_result['TRA_auc'],ALL_High_result['TRA_auc'],ALL_Low_result['TRA_auc']])

# Save results to csv files
if not os.path.exists(os.path.join(result_csv_name)):
    columns = ["High_threshold","Low_threhsold","Threshold"]
    columns.extend(["GT_ROT_acc","High_ROT_acc","Low_ROT_acc","GT_TRA_acc","High_TRA_acc","Low_TRA_acc"])
    columns.extend(["GT_ROT_kappa","High_ROT_kappa","Low_ROT_kappa","GT_TRA_kappa","High_TRA_kappa","Low_TRA_kappa"])
    columns.extend(["GT_ROT_auc","High_ROT_auc","Low_ROT_auc","GT_TRA_auc","High_TRA_auc","Low_TRA_auc"])
    pd.DataFrame(columns=columns).to_csv(result_csv_name,index=False)

df = pd.read_csv(result_csv_name)
df.loc[len(df)] = all_data
df.to_csv(result_csv_name,index=False)

for ind,column_name in enumerate(df.columns.values):
    print(column_name,df.values[len(df)-1][ind])




















