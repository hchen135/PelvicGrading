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
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
from util import clean_result,r_t_tile_form,df_gene,GT_label_to_dict,threshold_set,train_val_subject_gene,predict_rt,refine_predictions,fracture_avg_status
np.random.seed(1234)

parser = argparse.ArgumentParser()

parser.add_argument('--GT_FRAC_PATH',dest='GT_FRAC_PATH',type=str,default='/Users/hc/Documents/JHU/PJ/Mathias/AnnaProject/Pelvic/Journal_Extention/pelvic/GT/GT_dict.json')
parser.add_argument('--GT_LABEL_PATH',dest='GT_LABEL_PATH',type=str,default='/Users/hc/Documents/JHU/PJ/Mathias/AnnaProject/Pelvic/Journal_Extention/pelvic/GT/Automated Tile cases_Consensus_final consensus grades.csv')
parser.add_argument('--RESULT_PATH',dest='RESULT_PATH',type=str,default='/Users/hc/Documents/JHU/PJ/Mathias/AnnaProject/Pelvic/Journal_Extention/pelvic/final_results/out.json')
parser.add_argument('--high_threshold',dest='high_threshold',type=float,default=0.8)
parser.add_argument('--low_threshold',dest='low_threshold',type=float,default=0.5)
parser.add_argument('--threshold',dest='threshold',type=float,default=0.5)
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
ALL_Refine_result = {'ROT_acc':[],'ROT_auc':[],'ROT_kappa':[],'TRA_acc':[],'TRA_auc':[],'TRA_kappa':[]}

high_frac_prob = []
high_frac_y = []

low_frac_prob = []
low_frac_y = []

refinement_frac_prob = []
refinement_frac_y = []


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
            #origin="expert",
        )

    bn = BayesianNetwork(sm_manual)
    # First, learn all the possible states of the nodes using the whole dataset
    bn.fit_node_states(train.drop(columns=["name"]))
    # Fit CPDs using the training dataset
    bn.fit_cpds(train.drop(columns=["name"]), method="BayesianEstimator", bayes_prior="K2")
    # bn.fit_cpds(train.drop(columns=["name"]), method="MaximumLikelihoodEstimator")

    ie = InferenceEngine(bn)


    test=test.drop(columns=['tile'])
    test_high_conf=test_high_conf.drop(columns=['tile'])
    test_low_conf=test_low_conf.drop(columns=['tile'])

    print(test_high_conf)
    
    print("Ground Truth:")
    predict_rt(bn, test, ALL_GT_result)
    print()

    print("Predictions on High-confidence dictionary:")
    predict_rt(bn, test_high_conf, ALL_High_result)
    print()


    print("Predictions on Low-confidence dictionary:")
    predict_rt(bn, test_low_conf, ALL_Low_result)
    print()

    print("Take ONE iteration and refine predictions w help of low-confidence dictionary:")
    refined = refine_predictions(test_high_conf, test_low_conf, ie, args.threshold, 1)
    predict_rt(bn, refined, ALL_Refine_result)
    print()


    fracture_avg_status(high_frac_prob,high_frac_y,test_high_conf,RESULT_dict,GT_dict)
    fracture_avg_status(low_frac_prob,low_frac_y,test_low_conf,RESULT_dict,GT_dict)
    fracture_avg_status(refinement_frac_prob,refinement_frac_y,refined,RESULT_dict,GT_dict)

fpr, tpr, thresholds = roc_curve(high_frac_y, high_frac_prob)
plt.plot(fpr,tpr,label='Average ROC curve for high-conf threshold (area = 0.784)')
print('high auc:',roc_auc_score(high_frac_y,high_frac_prob))

# fpr, tpr, thresholds = roc_curve(low_frac_y, low_frac_prob)
# plt.plot(fpr,tpr,label='low')

fpr, tpr, thresholds = roc_curve(refinement_frac_y, refinement_frac_prob)
plt.plot(fpr,tpr,label='Average ROC curve for BM refinement (area = 0.809)')
print('refinement auc:',roc_auc_score(refinement_frac_y,refinement_frac_prob))

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("Average ROC curve for fracture detection")
plt.legend(loc='lower right')
plt.savefig("fracture_auc_avg.png")














