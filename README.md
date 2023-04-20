# Predicting Age Group Based on Image

# Goal
My goal is to make an image classification model to be able to predict well if a person is young or not, but I also want to predict well on the middle and old ages. I want to create visualizations and reports to see how well my model predicts on each class and how well it learns off the train data.

![64591966_605](https://user-images.githubusercontent.com/122238220/233473574-f7699007-ee88-4429-b4a6-43c5f7041955.jpg)

# Business Understanding

 My stakeholder is Tik Tok and Tik Tok has an issue with children lying about their age to not be as limited with their account which can lead to them getting advertisments that are not targeted towards them and inappropriate content that is not meant to be seen at their age. Recently Congress brought up this issue of inappropriate content being targeted towards children, but it is not necessarily entirely Tik Tok's fault. Part of the issue, as I mentioned before, is children lying about their when they make an account on the platform. Not only does Congress have an issue with this, but parents of the children are concerned for their kids being exposed inappropriate content that they themselves cannot control. Tik Tok needs a solution to this issue so I believe having an age image classification can be a right step in the right direction. Here is an article mentioning the issue Tik Tok is facing with parents concerned about their kid being exposed to inappropriate content on the platform.
 
 Article link: https://www.internetmatters.org/hub/esafety-news/tik-tok-app-safety-what-parents-need-to-know/
 
 # Data Understanding 
 
 I got my data from Kaggle which includes over 23,000 images of verying ages of people from 1 year olds to over 100 year olds. I am predicting on age and the target has three classes which are young age(ranging from 1 year olds to 29 year olds), middle age(ranging 30 year olds to 59 year olds), and old age(ranging from 60 year olds and over). I first lowered my data from over 23,000 images to over 3,000 images because 23,000 images would have taken too long to process which I did not have the time for. I balanced my classes through the OS when I lowered the images from over 23,000 to over 3,000 images and I visualized it to make sure it was actually balanced.
 
 ![Countplot](https://user-images.githubusercontent.com/122238220/233479599-1486a3ea-ab71-4a2d-83e7-7d434b48da85.jpg)

After I checked to see if I balanced my dataset I did a train, validation, test split with all three class subfolders in each folder. After I did a train, test, validation split I went into the preprocessing and creating an image data generator. You can see my train, validation, test split in my base model folder

Base model folder: https://github.com/srinir301/Age-Multiclass-Classification/blob/main/Train%2C%20Validation%2C%20Test%20Split.ipynb

# Data Preparation

For my data preparation, I started out with a face detection because I wanted to crop only the face to get rid of the noise in the background and it makes the model less complex. I got an OpenCV with a face detection algorithm on GitHub which is a pretrained algroithm that will detect faces on my images and the link to the OpenCV on Github is below.

Face Detection Algrorithm on GitHub: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

After I got the pretrained face detection I then adjusted the height and width of the images to get the face and some parts of the hair. I then made these cropped images as separate files and, like the original train, validation, test split, I made another split with the three classes as subfolders in each of the folders so that I can put it into my image data generator. After doing the face detection, I did an edge detection to try getting information such as the shape of the face and the wrinkles of the face which can help detect age. I used Sobel instead of Canny because Sobel gets more detail from the face than Canny. I then realized I came accross two problems:

1. There are alot of images that are blurry and the edge detection has trouble detecting certain areas

2. The Sobel edge detection leaves out information that is important for the model which is why I got a lower accuracy score than before

After I saw the issue with edge detection I thought I could try to sharpen the blurred out images so that the edge detection can get all the edges in the image, but sharpening the image still made the image look blurry. The pixels in the image did get smoothed out, but the image still looks bad. After I decided to ditch the edge detection and sharpening the images I did image augmentation to add more images for the model to train on, but I did not use image augmentation for my VGG19 and ResNet50 models because it actualy lowered my accuracy scores for those models. I did use augmentation for my CNN models which helped improve the accuracy score a bit. I then did transfer learning for my VGG19 and ResNet50 models by putting in specific pretrained preprocessing to the models. Here are some images showing my results of the edge detection and sharpening the images:

### Clear Image with Edge Detection


![Edge Detection](https://user-images.githubusercontent.com/122238220/233486346-8134c7be-8d10-4b63-83e0-b9efb00b16fd.jpg)


### Unclear Image with Edge Detection


![Edge Detection 2](https://user-images.githubusercontent.com/122238220/233486457-8ee34624-01f2-47cd-a87d-8c1b67bc7335.jpg)


### Sharpened Images


![Sharpened image](https://user-images.githubusercontent.com/122238220/233486526-9f6906ec-ae68-4452-8628-87774aef96e0.jpg)


# Modeling

I mainly created 3 models which were:

1. CNN Models

2. VGG19 Models

3. ResNet50 Models

I started out with a base CNN model with no preprocessing that got a test accuracy score of 48% which is not a bad start. That CNN model was the best score I got for the CNN models, but the that model was overfit which I needed to fix with regualrization and dropout. Then I made VGG19 models and my best VGG19 model got an accuracy score 76%. Lastly, I made ResNet50 models and my best ResNet50 model had an accuracy score of 55% which was a downgrade. My best model was the VGG19 model with an accuracy score of 76% and it was not overfit. Here are my confusion matrix for each of the models:


### Base CNN Model


![Base CNN model](https://user-images.githubusercontent.com/122238220/233489177-5c35218f-9837-46f7-aaea-660d5a6d0918.jpg)


### VGG19 Model


![New vgg19 model](https://user-images.githubusercontent.com/122238220/233489251-e87b04ce-8068-4629-b965-f8e385f739bc.jpg)


### ResNet50 Model


![resnet50 model](https://user-images.githubusercontent.com/122238220/233489305-a2aadb14-9394-4c1f-b019-b9077532724c.jpg)


# Evaluation

My best model, VGG19 model, was pretty simple compared to the other model because it only had one dense layer with:

- 256 nodes 

- L2 Regularization(0.001)

- Dropout(0.5)

- Early Stopping

- Reduce LROn Plateau

The Reduce LROn Plateau reduces the learning rate if the accuracy of the model does not change too much and and converges faster which improves the accuracy score. The dropout and regularization helps the model not overfit. Here is the ROC curve for the accuracy and the loss of the VGG19 model:


### Train and Validation Accuracy


![Accuracy](https://user-images.githubusercontent.com/122238220/233490846-7b4b4097-7cb4-4d71-a480-708739026ea8.jpg)


### Train and Validation Loss


![Loss](https://user-images.githubusercontent.com/122238220/233491074-eeaea70d-439d-4e37-b712-6b8e18059cf8.jpg)


# Recommendations

1. Tik Tok can take this model and make it a part of the verification process where users can take a picture of themselves and verify if they actually are that old. This is a starting point, but this should not be only way to verify the age and there should be other ways to verify as well

2. This model is not the best to use when the images are blurry because there are certain details that are left out and the model cannot read the image as well as a clear image

# Next Steps

1. I want to try sharpening or make the images at a higher quiality which can probably help improve my accuracy score

2. I want to deploy this model to see how it fairs against new images

3. I want to predict on 10 classes(which would be in the range of 10 years for each class) instead of 3 because I want to get more specific content and advertisements for each decade

4. Lastly, I want to try improving the accuracy of the model which I can probably do by trying a vgg16 model or add more and new layers


# Directories to Files in Repo

Presentation Slides: https://github.com/srinir301/Age-Multiclass-Classification/blob/main/Age%20Image%20Classification%20Project.pdf

Image Data: https://github.com/srinir301/Age-Multiclass-Classification/tree/main/Face%20Pictures

Final Notebook: https://github.com/srinir301/Age-Multiclass-Classification/blob/main/Final_Notebook.ipynb

Train, Validation, Test, Split Through OS: https://github.com/srinir301/Age-Multiclass-Classification/blob/main/Train%2C%20Validation%2C%20Test%20Split.ipynb

ReadMe: https://github.com/srinir301/Age-Multiclass-Classification/blob/main/README.md
