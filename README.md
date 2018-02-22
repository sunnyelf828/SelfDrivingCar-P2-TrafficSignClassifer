# SelfDrivingCar-P2-TrafficSignClassifer
### Summary

This Project used deep neural networks and convolutional neural networks to classify traffic signs. 

Recognition of traffic signs is a challenging real-world problem of high industrial relevance. Traffic sign recognition is a multi-class classication problem with unbalanced class frequencies. Traffic signs can provide a wide range of variations between classes in terms of color, shape, and the presence of pictograms or text. However, there exist subsets of classes (e. g., speed limit signs) that are very similar to each other.The classier has to cope with large variations in visual appearances due to illumination changes, partial occlusions, rotations, weather conditions, etc.

Specifically, I'll train a model to classify traffic signs from the **German Traffic Sign Dataset** which is large and lifelike — contains **43 classes of traffic sign**, with **more than 50,000 images in total**.  The deep neural network model is trained with **Tensorflow on AWS EC2**.  The model is modified on the basis of the famous 5-layer LeNet implementation. 

Before the model is trained, as the distribution of traffic sign classes are biased, **augment** is needed to improve the model performance. Common data augmentation techniques include rotation, translation, zoom, flips, and/or color perturbation. These techniques will be used individually or combined.

As a first step, I **convert the images to grayscale** to reduce the computational and storage cost while retain most features of traffic sign. In addtion, I **normalize the image** data to increase the performance of the model, making all the pixel value within the range of [0.1,0.9].

Then, the data are split into training set and test set and are trained in a **6 layer CNN model**. The model architecture is shown as below:

|                                       |                                               |
| ------------------------------------- | --------------------------------------------- |
| Layer                                 | Description                                   |
| Input                                 | 32x32x1 Grayscale image                       |
| **Layer 1**: Convolution 5x5          | 1x1 stride, valid padding, outputs 28x28x10   |
| RELU                                  | activation                                    |
| Max pooling                           | 2x2 stride, valid padding, outputs 14x14x10   |
| **Layer 2**: Convolution 3x3          | 1x1 stride, valid padding, outputs 12x12x40   |
| RELU                                  | activation                                    |
| Dropout                               | **avoid overfitting**, keep probability = 0.6 |
| **Layer 3**: Convolution 3x3          | 1x1 stride, valid padding, outputs 10x10x80   |
| RELU                                  | activation                                    |
| Max pooling                           | 2x2 stride, valid padding, outputs 5x5x80     |
| Flatten                               | outputs 2000                                  |
| **Layer 4**: Fully Connected          | outputs 120                                   |
| RELU                                  | activation                                    |
| **Layer 5**: Fully Connected          | outputs 84                                    |
| RELU                                  | activation                                    |
| **Layer 6**: Fully Connected —>logins | outputs 43                                    |

To train the model, I used the follow global parameters:

- Number of epochs = 10. Experimental way: increasing of this parameter doesn't give significant improvements.
- Batch size = 128
- Learning rate = 0.001
- Optimizer - Adam algorithm (alternative of stochastic gradient descent). Optimizer uses backpropagation to update the network and minimize training loss.
- Dropout = 0.6 (for training set only)

The model is validated and evaluated through calculating the accuracy on training set and validation set.A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

Once a final model architecture is selected, the accuracy on the test set will be calculated and reported as well.

The model is then used to test on the new images and predict the sign type. Since sometimes there might be false predictions, the softmax probability for the five candidates will be given and the correct classification is usually within these five candidates.



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report



#### Writeup / README

##### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sunnyelf828/SelfDrivingCar-P2-TrafficSignClassifer/blob/master/Traffic_Sign_Classifier.ipynb)

#### Data Set Summary & Exploration

##### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The data was loaded using pickle. I used the numpy library to calculate summary statistics of the trafficsigns data set:

- The size of training set is 34799
- The size of test set is 12630
- The shape of a traffic sign image is 32*32*3
- The number of unique classes/labels in the data set is 43

##### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed. It can be seen that the traffic signs are not uniformly distributed — **Augment** is needed to improve model performance. Common data **_augmentation techniques include rotation, translation, zoom, flips, and/or color perturbation_**. These techniques will be used individually or combined.



![output_8_0](/Users/chenm/Documents/Udacity_SelfDrivingCar/CarND-P2-Traffic-Sign-Classifier/output_8_0.png)

#### Design and Test a Model Architecture

##### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Before the model is trained, as the traffic sign classes are biased, **augment** is needed to improve the model performance. Common data augmentation techniques include rotation, translation, zoom, flips, and/or color perturbation. These techniques will be used individually or combined.

As a first step, I **convert the images to grayscale** because in case with traffic signs, the color is unlikely to give any performance boost. Then, I **normalize the image** data to reduce the number of shades to increase the performance of the model.

##### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

3 Convolutional layers with 3 fully connected layers. More can be seen from the above table.

##### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the follow global parameters:

- Number of epochs = 10. Experimental way: increasing of this parameter doesn't give significant improvements.
- Batch size = 128
- Learning rate = 0.001
- Optimizer - Adam algorithm (alternative of stochastic gradient descent). Optimizer uses backpropagation to update the network and minimize training loss.
- Dropout = 0.75 (for training set only)

##### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

- training set accuracy of 0.998

- validation set accuracy of 0.966

- test set accuracy of 0.944

  ​

#### Test a Model on New Images

##### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

​    

The first image might be difficult to classify because ...

##### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                | Prediction           |
| -------------------- | -------------------- |
| Traffic Sign         | Traffic Sign         |
| Speed limit (30km/h) | Speed limit (30km/h) |
| Road work            | Road work            |
| Turn right ahead     | Turn right ahead     |
| No entry             | No entry             |

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 



##### Discuss how you reach to the final model

starting with the neural network same with the LeNet

with rate = 0.01, EPOCH = 20

with rate = 0.001, EPOCH = 20

Add another Conv Layer with dropout, and make the depth deeper for each Conv Layer:

with rate = 0.001,EPOCH = 40, the validation accuracy is still below 0.9

change the normalization method:

from 

(image_data-128)/128

to [restricted to be within 0.1~0.9]

0.1+(image_data-0)*(0.9-0.1)/(255-0)

and the accuracy greatly improved:

with rate = 0.001,EPOCH = 20, the validation accuracy is nearly 0.95