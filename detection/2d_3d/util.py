import pandas as pd

def IOU_2d(loc1,loc2):
	assert len(loc1) == len(loc2),str(len(loc1))+"/"+str(len(loc2))
	assert len(loc1) == 4

	total = (loc1[2] - loc1[0])*(loc1[3]-loc1[1]) + (loc2[2] - loc2[0])*(loc2[3]-loc2[1])

	a1 = max(loc1[0],loc2[0])
	a2 = max(loc1[1],loc2[1])
	a3 = min(loc1[2],loc2[2])
	a4 = min(loc1[3],loc2[3])

	if a3 - a1 <= 0 or a4 - a2 <= 0:
		return 0.0

	sub = (a3 - a1)*(a4 - a2)
	return sub/(total - sub)

def IOU_3D(loc1,loc2):
    #print(loc1,"---",loc2)
    assert len(loc1) == len(loc2),str(len(loc1))+"/"+str(len(loc2))
    assert len(loc1) == 6
    
    total = (loc1[3] - loc1[0])*(loc1[4]-loc1[1])*(loc1[5]-loc1[2]+1) + (loc2[3] - loc2[0])*(loc2[4]-loc2[1])*(loc2[5] - loc2[2]+1)
    
    if loc1[3] < loc2[0] or loc2[3] < loc1[0] or loc1[4] < loc2[1] or loc2[4] < loc1[1] or loc1[5] < loc2[2] or loc1[5] < loc2[2]:
        return 0.0

    sub = (min(loc1[3],loc2[3]) - max(loc1[0],loc2[0]))*(min(loc1[4],loc2[4]) - max(loc1[1],loc2[1]))*(min(loc1[5],loc2[5]) - max(loc1[2],loc2[2])+1)
    return  sub/(total - sub) 

def bbox_enlarge_2d(record,new_values,confidence):
	loc = record["loc"]
	loc[0] = min(loc[0],new_values[0])
	loc[1] = min(loc[1],new_values[1])
	loc[2] = max(loc[2],new_values[2])
	loc[3] = max(loc[3],new_values[3])

	record["confidence"] = max(record["confidence"],confidence)


def bbox_enlarge_3d(record,new_values,slice_ind,confidence):
	loc = record["loc"]
	loc[0] = min(loc[0],new_values[0])
	loc[1] = min(loc[1],new_values[1])
	loc[3] = max(loc[3],new_values[2])
	loc[4] = max(loc[4],new_values[3])
	loc[5] = slice_ind

	record["confidence"] = max(record["confidence"],confidence)

def bbox_enlarge_3d_box(record,new_bbox):
    record["loc"][0] = min(record["loc"][0],new_bbox["loc"][0])
    record["loc"][1] = min(record["loc"][1],new_bbox["loc"][1])
    record["loc"][2] = min(record["loc"][2],new_bbox["loc"][2])
    record["loc"][3] = max(record["loc"][3],new_bbox["loc"][3])
    record["loc"][4] = max(record["loc"][4],new_bbox["loc"][4])
    record["loc"][5] = max(record["loc"][5],new_bbox["loc"][5])

    record["confidence"] = max(record["confidence"],new_bbox["confidence"])



def insert_2d(dict_2d,info_slice):
	img_name,x1,y1,x2,y2,label,confidence = info_slice
	x1 = int(x1)
	y1 = int(y1)
	x2 = int(x2)
	y2 = int(y2)
	label = int(label)
	confidence = float(confidence)

	img_subject = img_name.split(' ')[0]
	img_ind = int(img_name.split(' ')[1].split('.')[0])
	if img_subject not in dict_2d:
		dict_2d[img_subject] = {}
		dict_2d[img_subject]["z_min"] = img_ind
		dict_2d[img_subject]["z_max"] = img_ind
		dict_2d[img_subject]["bboxes"] = {}

	dict_2d[img_subject]["z_min"] = min(int(dict_2d[img_subject]["z_min"]),int(img_ind))
	dict_2d[img_subject]["z_max"] = max(int(dict_2d[img_subject]["z_max"]),int(img_ind))
	if img_ind not in dict_2d[img_subject]["bboxes"]:
		dict_2d[img_subject]["bboxes"][img_ind] = [{"loc":[x1,y1,x2,y2],"label":label,"confidence":confidence}]
	else:
		_processed = False
		for record in dict_2d[img_subject]["bboxes"][img_ind]:
			if record["label"] == label:
				if IOU_2d(record["loc"],[x1,y1,x2,y2]) > 0:
					bbox_enlarge_2d(record,[x1,y1,x2,y2],confidence)
					_processed = True
					break
		if not _processed:
			dict_2d[img_subject]["bboxes"][img_ind].append({"loc":[x1,y1,x2,y2],"label":label,"confidence":confidence})

def dict_3d_count_bbox(dict_3d):
    total_count = 0
    for img_subject in dict_3d:
        total_count += len(dict_3d[img_subject]["bboxes"])
    return total_count

def dict_3d_merge_box(dict_3d,threshold=0.5):
     
    for img_subject in dict_3d:
        processed = True
        while processed:
            new_bboxes = []
            processed = False
            for bbox in dict_3d[img_subject]["bboxes"]:
                tmp_processed = False
                for new_bbox in new_bboxes:
                    if new_bbox['label'] == bbox['label'] and IOU_3D(new_bbox["loc"],bbox["loc"]) > threshold:
                        bbox_enlarge_3d_box(new_bbox,bbox)
                        tmp_processed = True
                        break
                processed = processed or tmp_processed
                if not tmp_processed:
                    new_bboxes.append(bbox)
            if processed:
               dict_3d[img_subject]["bboxes"] = new_bboxes   
            


def dict_3d_construction(dict_3d,dict_2d,merge_3d3d_threshold=0.5):
    for img_subject in dict_2d:
        dict_3d[img_subject] = {"bboxes":[],"candidates":[]}
        for slice_ind in range(int(dict_2d[img_subject]["z_min"]),int(int(dict_2d[img_subject]["z_max"])+1)):
            if slice_ind not in dict_2d[img_subject]["bboxes"]:
                dict_3d[img_subject]["bboxes"].extend(dict_3d[img_subject]["candidates"])
                dict_3d[img_subject]["candidates"] = []
                continue
            for bbox in dict_2d[img_subject]["bboxes"][slice_ind]:
                x1,y1,x2,y2 = bbox["loc"]
                label = bbox["label"]
                confidence = bbox["confidence"]

                if len(dict_3d[img_subject]["candidates"]) == 0:
                    dict_3d[img_subject]["candidates"].append({"loc":[x1,y1,slice_ind,x2,y2,slice_ind],"label":label,"confidence":confidence})
                else:
                    #compare!
                    _processed = False
                    for candidate_box in dict_3d[img_subject]["candidates"]:
                        if candidate_box["label"] == label and IOU_2d(candidate_box["loc"][:2]+candidate_box["loc"][3:5],bbox["loc"]) > 0:
                            bbox_enlarge_3d(candidate_box,bbox["loc"],slice_ind,confidence)
                            _processed = True
                            break
                    if not _processed:
                        dict_3d[img_subject]["candidates"].append({"loc":[x1,y1,slice_ind,x2,y2,slice_ind],"label":label,"confidence":confidence})
            #After screening a slice, check if any bboxes are now out-dated
            old_candidates = dict_3d[img_subject]["candidates"]
            new_candidates = []
            for bbox in old_candidates:
                if bbox["loc"][-1] != slice_ind:
                    dict_3d[img_subject]["bboxes"].append(bbox)
                else:
                    new_candidates.append(bbox)
            dict_3d[img_subject]["candidates"] = new_candidates
        dict_3d[img_subject]["bboxes"].extend(dict_3d[img_subject]["candidates"])
        del dict_3d[img_subject]["candidates"]

    dict_3d_merge_box(dict_3d,merge_3d3d_threshold)

def dict_3d_to_csv(dict_3d,column_names = ["img_name","x1","y1","z1","x2","y2","z2","label","confidence"]):
	results = []
	for img_subject in dict_3d:
		for bbox in dict_3d[img_subject]["bboxes"]:
			x1,y1,z1,x2,y2,z2 = bbox["loc"]
			results.append([img_subject,x1,y1,z1,x2,y2,z2,bbox["label"],bbox["confidence"]])
	return pd.DataFrame(results,columns = column_names)

def dict_3d_counting_init(dict_3d_ori):
    for image_subject in dict_3d_ori:
        for bbox in dict_3d_ori[image_subject]["bboxes"]:
            bbox['valid_counting'] = 0
            bbox['total_counting'] = 0

def dict_3d_counting_increment(dict_3d_ori):
    for image_subject in dict_3d_ori:
        for bbox in dict_3d_ori[image_subject]["bboxes"]:
            bbox['total_counting'] += 1

def dict_3d_counting_update(dict_3d_ori,dict_3d_aug,threshold=0.0):
    dict_3d_counting_increment(dict_3d_ori)
    for image_subject in dict_3d_aug:
        for bbox_aug in dict_3d_aug[image_subject]["bboxes"]:
            for bbox_ori in dict_3d_ori[image_subject]["bboxes"]:
                # use sign to record whether this fracture has been found or not.
                if bbox_aug["label"] == bbox_ori["label"] and IOU_3D(bbox_ori["loc"],bbox_aug["loc"]) > threshold and bbox_ori['total_counting']>0:
                    #print(bbox_ori["loc"],"---",bbox_aug["loc"],"---",IOU_3D(bbox_ori["loc"],bbox_aug["loc"]))
                    bbox_ori["valid_counting"] += 1
                    bbox_ori['total_counting'] *= -1
        for bbox_ori in dict_3d_ori[image_subject]["bboxes"]:
            bbox_ori['total_counting'] = abs(bbox_ori['total_counting'])





		



