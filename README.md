# Image-retrieval-system

In this project I created an image retrieval system that can browse, search and retrieve images from an image database

the system is basicly two phases :

## 1- Feature extraction 

in this phase I used the classification model resnet 50 pretrained on imagenet dataset without the last layer of it, because I want to use it as an image feature extractor to extract features from all images in the database and save those feature vectors

## 2- Similarty algorithm

to search among the images in the database, I first extract the feature vector from the desired image the same way, then I find the similar vector using a similarity algorithm, I used KNN - K nearest neighbor - as a similarity alforithm.
