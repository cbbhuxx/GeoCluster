# GeoCluser
Enhancing Visual Place Recognition in Spatial Domain on Aerial Vehicle Platforms.

| Aerial localisation
| :-------------------------:
| ![image](https://github.com/cbbhuxx/GeoCluster/blob/master/img/example_localisation.gif)

## Prerequisites
 - torch 
 - torchvision
 - opencv-python
 - tensorboard
 - matplotlib
 - sklearn
 - numpy

Please see "requirements.txt" for a detailed list of packages and versions.

## The code expects data to be in the following directory structure:
 ``` 
 experiment\
 ├── features
 |   ├── Beijing
 |   |   ├── Beijing_feature_1.00.npy
 |   |   ├── Beijing_feature_0.85.npy
 |   |   ├── Beijing_feature_0.85.npy
 ├── map
 |   ├── Beijing.tif   
 ├── query_images
 |   ├── Beijing_query
 |   |   ├── 0.png
 |   |   ├── 1.png
 ├── route
 |   ├──Beijing_route.npy
 ├── tiles
 |   ├── Beijing_ref_scale_0.85
 |   |    ├── 0.png
 |   |    ├── 1.png
 |   ├── Beijing_ref_scale_1.00
 |   |    ├── 0.png
 |   |    ├── 1.png
 |   ├── Beijing_ref_scale_1.35
 |   |    ├── 0.png
 |   |    ├── 1.png
 ``` 
   Note: scale_1.00, scale_0.85, scale_1.35 refer to the scale of the map tile image and the camera image.

## Quick start
 If there is no database, put the database image into experiment/tiles/:
```
python main.py --model=GeoCluster --mode=Pre
```
If the database already exists:
```
python main.py --model=GeoCluster --mode=PF
```

## License + attribution/citation
When using code within this repository, please refer the following [paper](https://xplorestaging.ieee.org/document/10423811) in your publications:
```
@ARTICLE{10423811,
  author={Chen, Chao and He, Mengfan and Wang, Jun and Meng, Ziyang},
  journal={IEEE Robotics and Automation Letters}, 
  title={GeoCluster: Enhancing Visual Place Recognition in Spatial Domain on Aerial Vehicle Platforms}, 
  year={2024},
  volume={9},
  number={3},
  pages={3013-3020},
  keywords={Feature extraction;Visualization;Task analysis;Training;Location awareness;Databases;Autonomous aerial vehicles;Vision-based navigation;localization;recognition;deep learning for visual perception},
  doi={10.1109/LRA.2024.3363536}}
```
