## Sagemaker-Pytorch-Image-Segmentation-with-Custom-Dataset
Sagemaker Pytorch Satellite Image Segmentation with Custom Dataset

# Road segmentation with Neural Networks

Aaron Kaefer

This project presents a solution to detecting roads from satellite images. The classifier consists of the fully Convolutional Neural Network DeeplabV3 which outputs for each pixel of an input image whether or not it is considered being part of a road.

The model is trained on more than 6000 satellite images obtain an accuracy of ~ 95% on the test set.

This project has been developed to demonstrate how custom image datasets can be used to train PyTorch models on Amazon Sagemaker. The code can be used as an example on how to create a custom PyTorch dataloader with the necessary dependencies to operate with Amazon Sagemaker.

Here is an example of the classfication result for two test images:

![palu_tsunami_aoi_pre_img_90](https://github.com/user-attachments/assets/bddf443d-8032-47a3-8e02-09a74f32a211)
![infh_palu_tsunami_aoi_pre_img_90](https://github.com/user-attachments/assets/4ef5e670-f0bb-42d3-b9d1-764c8d5216e9))

# Storing the custom dataset in Amazon S3, creating a custom Dataloader, and setting the correct file paths
The final processed dataset contains pairs of images and masks(labels). They are stored in an Amazon S3 bucket in the following hierarchy:

```
S3 Bucket
  |––Data
       |––Train data
       |    |––Train Images
       |    |––Train Masks
       |––Test data
            |––Test Images
            |––Test Masks
```
In this project the file hierarchy is named as follows:
```
Name of S3 Bucket
  |––sagemaker
       |––seg-data
            |––train
            |    |––images
            |    |––masks
            |––test
                  |––images
                  |––masks
```

Now lets pass the S3 directory where the training and test data is stored to our Sagemaker training job. This is done with the following code. The custom Dataloader is defined in the deeplab.py script.
<img width="986" alt="Screenshot 2024-10-22 at 19 05 38" src="https://github.com/user-attachments/assets/0aba9bf9-dd94-43b2-8687-eca73edece32">
<img width="981" alt="Screenshot 2024-10-22 at 18 34 27" src="https://github.com/user-attachments/assets/9108310e-12d1-465f-85af-dee803e0aa90">

Finally let's write the custom PyTorch Dataloader to handle the custom image dataset properly:
<img width="675" alt="Screenshot 2024-10-22 at 18 00 00" src="https://github.com/user-attachments/assets/e1ce711c-c52f-46e4-bfc5-2fc9f008ffb6">
<img width="749" alt="Screenshot 2024-10-22 at 18 00 47" src="https://github.com/user-attachments/assets/585d4336-5704-4a85-a4a6-1c10c110b3c8">
<img width="641" alt="Screenshot 2024-10-22 at 18 01 07" src="https://github.com/user-attachments/assets/b7aaf57d-f265-444e-b234-ab03cd8a7b67">







