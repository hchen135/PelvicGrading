# PelvicGrading
This repository is the code for paper "Interpretable Severity Scoring of Pelvic Trauma Through Automated Fracture Detection and Bayesian Inference".

The code has 2 parts: 
- fracture detection & 3D fracture fusing
- bayesian model classification

## Fracture Detection 
We use Faster RCNN with tensorflow to detect 2D fractures. Then we fuse 2D fractures into 3D ones with self-ensembling augmented inference introduced in the paper.

### Detection Model Training
First, extract 2D images from 3D CT scans and formulate them into 5-folds. Save images in each fold, such as location at `detection/fold3/train/images/`. Then create a csv file indicating the gound truth information. An example exists at `detection/fold3/train/cr3_fp.scv`.

Run `detection/train_rcnn.py` to train the model. It is required to train a model for each fold.

### 3D Fusion
We fuse 2D fractures into 3D ones and using self-ensembling augmented inference to calculate the frequency as the confidene of 3D fractures.

Run `detection/test_rcnn.py` to generate 2d detection results with different data augmentation.

Run `detection/2d_3d/2d_3d_convertion.py` to fuse 2D results into 3D ones.

Run `detection/2d_3d/3d_3d_counting.py` to calculate the frequency of 3D detected fractures among all augmentations. The output `out.json` will be the input of Bayesian models.

Here is an example of the structure of `out.json`:
```python
{
img_subject:
    {
    "bboxes":
        {[
            {
            "loc":[x_min,y_min,z_min,x_max,y_max,z_max],
            "label":label,
            "confidence":confidence
            },
            ...
        ]}
    }
}
```

### Bayesian models.
With the `out.json` generated from the previous step, we use `bayes/run.sh` to get the best results proposed in the paper.

To run the results without Bayesian refinement, we run `bayes/norefine.sh`. \
To run the results without age variable, we run `bayes/run_noage.sh`.

Results will be saved in `result_csv_name` in the corresponding python script.
