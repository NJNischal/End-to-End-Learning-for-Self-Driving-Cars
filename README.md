# End-to-End-Learning-for-Self-Driving-Cars

## Overview

This is a pytorch implementation of end to end learning for autonomous vehicles.


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
