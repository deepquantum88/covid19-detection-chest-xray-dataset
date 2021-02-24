# covid19-detection-chest-xray-dataset

# Overview of the Project

## Problem Statement - 
#### Detecting Covid19 from Chest X-ray images of patient using Quantum Neural Network

## Dataset used - 
We have used [this datset](https://www.kaggle.com/pranavraikokte/covid19-image-dataset) from Kaggle which contains 250 training and 65 testing images for our model. 
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/xray_example.jpeg?raw=true)


## Our Approach to the classifier- 

### Preprocessing the dataset
Images given in the dataset are real life chest x-rays and are not previouly modified. So all the images are of different dimensions. Intially, we reduced the image size to a specific dimension. Although, it is more convenient to fix the image size to 256x256, but due to limitations of computational resources, we can reduce it to 28x28 size. 
The dataset is provied in a folder format where all images of each classes are stored in different folders. We used [this python script](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/script_conv_to_csv.py) to convert those images to 28x28 using openCV library and finally saved in csv format. You can get here the [train csv](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/train.csv) and [test csv](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/test.csv). 


### Applying Quanvolutional Layer
We have extended the single layer approach of Quanvolutional Neural Network from [here](https://pennylane.ai/qml/demos/tutorial_quanvolution.html) to multiple layers, to be exact 4 layers in our model. You can get the notebook [here](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quanvolution_on_xray_image.ipynb). 

*In case notebook doesn't render properly, you can see this [pdf](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quanvolution_on_xray_image.pdf)

Initially, each image has the dimension of (28x28x1) which is further fed to the first Quanvolutional layer and converted to (14x14x4). The 2nd Layer converts it to (7x7x16), 3rd layer to (3x3x64) and finally the 4th and last layer converts each image to a (1x1x256) dimensional data matrix. 
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/qnn.png?raw=true)

### Classifier Model
After applying the Quanvolutional layers, we have employed the quantum classifier model. It consists of two subclassifiers each of which is a binary classifiernamed as 'Model-1' and 'Model-2', respectively.

Model-1 classifies between two classes - 'Normal Person' and 'Covid19/Viral Pnemonia'. 

Model-2 classifies between two classes - 'Covid10' and 'Viral Pneumonia'. 
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/models.png)

##### We have created two notebooks for this. 

In [Notebook-1](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quantum_classifier_1.ipynb) we have used 11 features from 256 feature sized input data. 
In [Notebook-2](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quantum_classifier_2.ipynb) we have reduced 256 features of each image to 4 using TruncatedSVD method. 

We have done this beacuse encoding 256 features to a quantum circuit is not a feasible approach. 

### Prediction
While predicting, we have given have input to the Model-1. If it predicts as Normal person, then it is the final prediction assigned to the input. If not, then we have given the same input to Model-2 and it finally predicts whether the chest x-ray is of Covid10 patient or of Viral Pneumonia patient.

### Plots for Trainging cost and accuracy for Model-1 and Model-2

#### Cost Plot for Model-1
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/cost_plot_model_1.png?raw=true)
#### Trainging accuracy plot for Model-1
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/train_acc_plot_model_1.png?raw=true)
#### Cost Plot for Model-2
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/cost_plot_model_2.png?raw=true)
#### Trainging accuracy plot for Model-2
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/train_acc_plot_model_2.png?raw=true)


# Some drawbacks in the draft due to lack of computational resources - 

1. The real x-ray images in the dataset are enough large to contain a lots of information. But due to lack of computational resources, we reduced the size to 28x28 using openCV library, which may have suppressed a lot of important informations. 
Later with the availability of more computational resources, we can use 256x256 dimensional image which will definitely increase the accuracy of the model. 

2. Currently, after applying Quanvolution and flattening the data we had 256 features of each image, we have used only 11 features by feature selection method due to lack of resources. We can try out with more features with availability of more qubits to experiment how accuracy imporves with more number of features. 

3. In this work, the whole training and prediction has been done on a simulator, not real Quantum computer (as it has been recommended not to train circuit on QC with that 250 USD credit), but we are really hopeful about some improvements if training will be executed on a real-time Quantum computer in future.

Now while training each model, number of times we call quantum circuit is equal to
 (shots assigned to the device * number of training images * number of iteration)
with this above rough calculation and as we have 250 training images, the resources required can be predicted easily.

4. All the four quanvolutional layers applied on the image data are using uniformly generated random parameters, which are not further trained. 

Later, we can experiment with training those quanvolutional layers too the way its has been done done in classical convolution.
