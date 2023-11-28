# Mushroom-Classification
This script describes the process of developing a deep learning model for classifying images of mushrooms. 
The primary focus is to train a convolutional neural network (CNN) using TensorFlow and Keras and to evaluate the model's performance through accuracy metrics and confusion matrices.

## Problem Solved:
The task was to build a CNN capable of identifying and classifying mushroom images into their respective genera based on visual features. 
The goal was to achieve high accuracy in classification to assist in mycological studies or potentially aid in distinguishing edible from poisonous mushrooms.

## Methods Used:
Data preprocessing included resizing images to a uniform (64x64) format for consistent input into the model.
Constructed a CNN model with multiple layers, including convolutional layers, max pooling, and dense layers for feature extraction and classification.
Implemented data augmentation techniques like horizontal flips, rotations, zooms, and shears to increase the diversity of training data and improve the model's generalization ability.
Utilized TensorFlow's built-in functions to cache, shuffle, and prefetch the dataset to optimize the training process.
Normalized the images by scaling the pixel values to the range [0,1].
Trained the model using the Adam optimizer and sparse categorical crossentropy loss function, over a specified number of epochs and with a set batch size.
## Results:
The training and validation processes were visualized through plots showing accuracy and loss over the epochs. 
The model's performance was quantitatively assessed using a confusion matrix, which provided insights into the true positive and false positive rates for each class. 
The model achieved the following results on the validation dataset:
Correctly identified 79.81% of 'Boletus' and 81.88% of 'Lactarius', indicating a well-performing model in terms of classification accuracy.
These results can be added to a resume to demonstrate hands-on experience with image classification using CNNs and TensorFlow, highlighting technical skills in deep learning, data preprocessing, and model evaluation.
