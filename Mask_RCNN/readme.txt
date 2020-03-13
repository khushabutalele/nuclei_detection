Mask R-CNN for nuclei detection and Segmentation

This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. The trained model generates segmentation masks for each instance of each nuclei in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone network for symantic segmentaion.

This assignment includes:

1.    Source code of Mask R-CNN built on FPN and ResNet50
2.    Training code for nuclei detection for your dataset
3.    Pre-trained weights for nuclei detection.
4.    ParallelModel class for multi-GPU training
5.    Example of training on your own dataset
6.    Sample images stored in images folder
7.    Detection is stored in result/nucleus folder
8.    mAP (mean average predition) screenshot is stored in folder mAP


Installation :-

1. CUDA 10.0 needed.
2. python3.5 and above versions are supported.
3. other necessary dependancies can be installed using - 
        pip3 install -r requirements.txt
4. if you dont have ROS installed in your machine, please comment line no. 28 in sample/nucleus/nucleus.py which is -- sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
   

"""
to run prediction

 python3 samples/nucleus/nucleus.py detect --dataset=/path/to/dataset  --subset=path/to/your/folder --weights=/model/mask_rcnn_nucleus_0021.h5 


"""


Usage: Run from the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier. you can use last if you trained model using your dataset so that it can take latest one.
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last




