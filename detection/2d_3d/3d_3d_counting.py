import argparse
import json
import numpy
import os
from glob import glob

from util import dict_3d_counting_init, dict_3d_counting_update

DIR_ori = "../3d_ori"
DIR_aug = "../aug"
DIR_out = "../out"
threshold = 0.5

#ori_json_name = glob(os.path.join(DIR_ori,"*.json"))[0]
ori_json_name = os.path.join(DIR_aug,"IntensityShift0_IntensityScale1.0_ContrastScale1.0.json")
aug_json_name_list = glob(os.path.join(DIR_aug,"*.json"))

with open(ori_json_name) as a:
    dict_3d_ori = json.load(a)

dict_3d_counting_init(dict_3d_ori)

_count = 0
for aug_json_name in aug_json_name_list:
    with open(aug_json_name) as a:
        dict_3d_aug = json.load(a)
    dict_3d_counting_update(dict_3d_ori,dict_3d_aug,threshold)
    print("Finished",_count,":",aug_json_name)
    _count+=1

with open(os.path.join(DIR_out,"out.json"),"w") as a:
    json.dump(dict_3d_ori,a,indent=4)

