"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    utils.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Supplemetary file to run the train file containing the functions to 
  *   augment and preprocess the images and steering angles.  
 """
import cv2, os
import numpy as np
import matplotlib.image as mpimg

# declaration of image parameters
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

"""
* @brief Function to load the images from the path 
* @param data direcrtory 
* @param File path of the images
* @return The image file
"""
def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

"""
* @brief Function to crop the imeages to the required shape
* @param The image to crop
* @return The cropped image
"""
def crop(image):
   
   # Crop the image (removing the sky at the top and the car front at the bottom)
    
    return image[60:-25, :, :] # remove the sky and the car front

"""
* @brief Resize the image to the input shape used by the network model
* @param Image file to Resize
* @return The Resized image for the network
""" 
def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

"""
* @brief Fuinction to convert the color space of the image from RGB to YUV.
* @param The image to change the colorspace.
* @return The converted image. 
"""
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

"""
* @brief Function to run the preprocess of the images.
* @param The images from the dataset
* @return The preprocessed images.
"""
def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

"""
* @brief Function to randomly choose the center, left and right images 
* to adjust the steering angles.
* @param Steering angles that corresponds to the images.
* @return The new adjusted steering angles.
"""
def choose_image(steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return "img_left_pth", float(steering_angle) + 0.2
    elif choice == 1:
        return "img_right_pth", float(steering_angle) - 0.2
    return "img_center_pth", float(steering_angle)

"""
* @brief Function to randomly flip the left and right images and adjust
* the steering angles.
* @param The images from the left or right dataset.
* @param The steering angle of the corresponding images.
* @return the flipped images and their corresponding steering angles.
"""
def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

"""
* @brief Fuinction to randomly translate the image vertically and horizontally.
* @param The image to translate.
* @param The steering angle of the selected image.
* @param Range of x translation.
* @param Range of y translation.
* @return The image and its corresponding steering angle
"""
def random_translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

"""
* @brief Randomly generates shadows in the image.
* @param The image to add shadow on.
* @return The image with added shadows.
""" 
def random_shadow(image):
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

"""
* @brief Function to adjust the brightness of the images.
* @param The image to process
* @return The brightness equalized image.
"""
def random_brightness(image):
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

"""
* @brief Function to augment the image and trhe steering angle of the center image
* @param The data directory containing thje images.
* @param The center, left and right images and their corresponding steering angles.
* @param The x and y range of the image augmentation allowd(set to a default value
* unless specified on execution)
* @return Returns the output from all image augmentation process.
"""
def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

"""
* @brief Function to Generate training image give image paths and associated steering angles
* @param The directory path where the data is stored.
* @param The image paths for each image.
* @param The steering angles for the images.
* @param The variable containing the size of the required batch size.
* @param Binary value of the status of the training process.
* @return The images and steering angles.
"""
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
