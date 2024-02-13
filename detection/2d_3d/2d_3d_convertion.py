import argparse
import pandas as pd
# import numpy
import os 
from util import insert_2d,dict_3d_construction,dict_3d_to_csv
import json
from glob import glob 

NUM_FOLD = 5
merge_3d3d_threshold = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--phase',default='ori',type=str,dest='phase',
                    help='whether the original prediction (ori) or augmented prediction (aug)')

args = parser.parse_args()
phase = args.phase

if phase == 'ori':
    DIR_2D = "../../2d_results/2d_ori/"
    DIR_OUT = "../3d_ori"

    candidate_prefix = ['bb_tr']
    candidate_suffix = ['_fp.csv']
    out_name = "3d_pred.csv"
elif phase == 'aug':
    DIR_2D = "../../2d_results/aug/"
    DIR_OUT = "../aug"

    candidates = set()
    for name in glob(os.path.join(DIR_2D,"*.csv")):
        candidates.add('_'+'_'.join(name.split('/')[-1].split('_')[1:]))
    candidate_suffix = list(candidates)
    candidate_prefix = ['fold']*len(candidate_suffix)

sorted(candidate_suffix)
print("Total subjects:",len(candidate_suffix))

if not os.path.exists(DIR_OUT):
	os.mkdir(DIR_OUT)
column_names = ["img_name","x1","y1","x2","y2","label","confidence"]

# {"imgt100":{"z_min":,"z_max":,"bboxes":{"44":[{"loc":[1,2,3,4],"label":class,"confidence":pred}]}}}


for candidate_ind in range(len(candidate_prefix)):
    dict_2d = {}
    for i in range(NUM_FOLD):
        info = pd.read_csv(os.path.join(DIR_2D,candidate_prefix[candidate_ind]+str(i)+candidate_suffix[candidate_ind])).values
        for pred_ind in range(info.shape[0]):
            insert_2d(dict_2d,info[pred_ind])
    # print(json.dumps(dict_2d,indent=4))
    dict_3d = {}
    dict_3d_construction(dict_3d,dict_2d,merge_3d3d_threshold)
    # print(dict_3d)
    df_3d = dict_3d_to_csv(dict_3d)

    if phase == 'aug':
        out_name = candidate_suffix[candidate_ind][1:]
    df_3d.to_csv(os.path.join(DIR_OUT,out_name),index=False)
    with open(os.path.join(DIR_OUT,".".join(out_name.split('.')[:-1])+'.json'),'w') as a:
        json.dump(dict_3d,a,indent=4)
    print("Finished:", candidate_ind, out_name)


