## Sagemaker-Pytorch-Image-Segmentation-with-Custom-Dataset
Sagemaker Pytorch Satellite Image Segmentation with Custom Dataset

# Road segmentation with Neural Networks

Aaron Kaefer

This project presents a solution to detecting roads from satellite images. The classifier consists of the fully Convolutional Neural Network DeeplabV3 which outputs for each pixel of an input image whether or not it is considered being part of a road.

The model is trained on more than 6000 satellite images obtain a decent accuracy of 94.5% on the test set.

This project has been developed to demonstrate how custom image datasets can be used to train PyTorch models on Amazon Sagemaker. The code can be used as an example on how to create a custom PyTorch dataloader with the necessary dependencies to operate with Amazon Sagemaker.

Here is an example of the classfication result for two test images:
