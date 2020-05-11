# End-to-End-Learning-for-Self-Driving-Cars

## Overview

This is a pytorch implementation of end to end learning for autonomous vehicles.

## Network Architecture
The design of the network is based on the model used by NVIDIA for the end-to-end self driving test.

### The model:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Flatten
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1 (output)

As per the NVIDIA model, the convolution layers are meant to handle feature engineering and the fully connected layer for predicting the steering angle. We have added the sensor datas from the Throttle, Brake and Speed to the network to enable autonomous driving.

The following shows the model with details on the shapes and the number of parameters.

| Layer                          |Output Shape      |Params  |Connected to     |
|--------------------------------|------------------|-------:|-----------------|
|lambda_1 (Lambda)               |(None, 90, 320, 3)|0       |lambda_input_1   |
|convolution2d_1 (Convolution2D) |(None, 43,158, 24)|1824    |lambda_1         |
|convolution2d_2 (Convolution2D) |(None, 20, 77, 36)|21636   |convolution2d_1  |
|convolution2d_3 (Convolution2D) |(None, 8, 37, 48) |43248   |convolution2d_2  |
|convolution2d_4 (Convolution2D) |(None, 6, 33, 64) |27712   |convolution2d_3  |
|convolution2d_5 (Convolution2D) |(None, 4, 33, 64) |36928   |convolution2d_4  |
|flatten_1 (Flatten)             |(None, 1152)      |0       |convolution2d_5  |
|dense_1 (Dense)                 |(None, 100)       |115300  |flatten_1        |
|dense_2 (Dense)                 |(None, 50)        |5050    |dense_1          |
|dense_3 (Dense)                 |(None, 10)        |510     |dense_2          |
|dense_4 (Dense)                 |(None, 1)         |11      |dense_3          |
|                                |**Total params**  |251709  |                 |


## How to Run

### Dependencies

You need to install all dependencies by installing the following packages

```
python3
numpy
matplotlib
jupyter
opencv
pillow
scikit-learn
scikit-image
scipy
h5py
eventlet
flask-socketio
seaborn
pandas
imageio
moviepy
tensorflow
pytorch
```

### Launch the autonomous car simulator:

1) Download and run the Udacity Self-Driving car simulator (https://github.com/udacity/self-driving-car-sim)
2) Select Track 1 or 2 and continue with either training or autonomous mode.

### Collecting the training data:

1) To train, select the 'Training mode' and click on record button or 'R' key on the keyboard.
2) Select the folder to save the data to and proceed with moving the vehicle with either the mouse, keyboard or a joystick.
3) To complete the collection of data press the 'R' key again. The file 'driving_log' will have the table of the data collected.

### Training the model:

To train the CNN model, run the following commands in the terminal:

```
python train.py -d <data directory>
```

In order to change the training parameters for the training process, you can change the parameters by using the following commands.

In order to change the epoch count:(Default is 10)
```
python train.py -d <data directory> -n <change to new epoch count>
```

In order to change the batch size:(Default is 40)
```
python train.py -d <data directory> -b <change to new batch size>
```

In order to change the learning rate:(Default is .001)
```
python train.py -d <data directory> -l <change to new learning rate>
```
This code checks for GPU and runs pytorch cuda if GPU is detected, else, the training takes place on the CPU.

The training took approximately 12 minutes per epoch when trained on a Nvidia GTX 1060 6GB GPU for a training set of around 200 training points (3 images each). Track 1 converges at 5 epochs while track 2 converges at 20 epochs.

### Testing the trained models

Multiple trained models will be stored in a folder called 'models' while training. We use these models to test the perfomance.
To test any model, launch the simulator, select the 'Autonomous mode' and then run the following command:
```

python drive.py <path to model_weights>

#Usage:

python drive.py models/model_19500

```
### Training Data if needed:

You have to change the path to the image directory in the driving_log.csv file to use this data to train the model. I have uploaded both the training data for both tracks.

```
https://drive.google.com/open?id=1nlyP8bC5arMCnqzGUZQVOtGIip23tWIA
```
