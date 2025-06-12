## PSO-HEAD: Pseudo-Supervision Guided Spatial Optimization for View-consistent 3D Full-Head Reconstruction<br>


![FIRST image](./misc/first.png)
<img src="./misc/post1.gif"  width="24%">
<img src="./misc/post2.gif"  width="24%">
<img src="./misc/post3.gif"  width="24%">
<img src="./misc/post4.gif"  width="24%">
<img src="./misc/post5.gif"  width="24%">
<img src="./misc/post6.gif"  width="24%">
<img src="./misc/post7.gif"  width="24%">
<img src="./misc/post8.gif"  width="24%">

**PSO-HEAD: Pseudo-Supervision Guided Spatial Optimization for View-consistent 3D Full-Head Reconstruction**<br>

## Requirements
```
cd pso-head
conda env create -f environment.yml
conda activate pso-head
```
## Obtaining camera pose and cropping the images

Please follow the [guide](Pose Estimator/cropping_guide.md)
### Steps


#### 1. cd to 3DDFA_V2. 
```.bash
cd 3DDFA_V2
```

#### 2. Extract face keypoints using dlib. 
```.bash
python dlib_kps.py 
```

#### 3. Obtaining camera poses and cropping the images using recrop_images.py

```.bash
python recrop_images.py -i data.pkl -j dataset.json
```

## Generate full head reconstruction from a single RGB image
```
.bash
# Please refer to ./gen_pti_script.sh
```

## Datasets
Due to the license issue, we are not able to release [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) and [K-Hairstyle dataset](https://psh01087.github.io/K-Hairstyle/) that we used to train the model. [test_data_img](./dataset/testdata_img/) and [test_data_seg](./dataset/testdata_seg/) are just an example for showing the dataset struture. For the camera pose convention, please refer to [PanoHead](https://sizhean.github.io/panohead). 


## Datasets format
For training purpose, we can use either zip files or normal folder for image dataset and segmentation dataset. For PTI, we need to use folder.
To compress dataset folder to zip file, we can use [dataset_tool_seg](./dataset_tool_seg.py). 

## Obtaining segmentation masks
You can try using deeplabv3 or other off-the-shelf tool to generate the masks.



## Acknowledgements

We thank Shuhong Chen for the discussion during Sizhe's internship.

This repo is heavily based off the [PanoHead](https://sizhean.github.io/panohead) repo; Huge thanks to the EG3D authors for releasing their code!
