# LYSTO-Challenge
Image Analysis

This project was carried out using the dataset providied by the Lymphocyte Assessment Hackathon (LYSTO). LYSTO is a challenge on assessing the number of lymphocytes in a given histopathology image. 

The method involves developing a machine learning model for analysing the given image to identify the corresponding number of cells (lymphocytes) in the image. The first step involves pre-processing of images. During pre-processing, the images are converted from RGB space to HED space, which typically involves changing the color space. The D channel in the new color space gives the stained form of the original image that can be used for identifying the number of lymphocytes present. 

The pre-processed images are spilt into training, validation, and test sets for model training and testing. Convolutional Neural Network (CNN) architecture is used for model construction. The model was trained and test for various hyper parameter values to get the best R2 score on the validation as well as the test data.

The model gives an R2 score of 0.8121, 0.7398, and 0.7332 on training, validation, and test data respectively.

The dataset can be downloaded from the following link - http://shorturl.at/fuCEO
