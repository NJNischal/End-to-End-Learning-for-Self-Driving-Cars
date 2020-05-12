 
"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    data_extractor.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Class declaration for extracting the data from csv file 
 """

import csv
import os
from torch.utils.data import Dataset
import cv2


"""
 * @brief The class Features is used to read and seperate various parameters from the 
 * csv file and return them as arrays for the train process.
 * @ param Gets the dataset directory path.
 """
class Features(Dataset):
    """
    * @brief Initializes the parameters in the Features class.
    * @param Path to the dataset.
    * @return None. 
    """
    def __init__(self, path_to_csv):
        #imports the dataset path and joins it to the csv log file.
        path_to_csv = os.path.join(path_to_csv, 'driving_log.csv')
        self.csv_data = self.load_csv_file(path_to_csv)
    """
    * @brief Function to load the csv file and seperate the values written in them.
    * @param Path to the csv file.
    * @return The split and extracted consolidated data.
    """
    def load_csv_file(self, path_to_csv):
        data = []
        with open(path_to_csv, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            for row in data_reader:
                data.append(row)
        return data
    """
    * @brief Function to test the return of csv data
    * @param None.
    * @return The csv data
    """ 
    def get_csv_data(self):
        return self.csv_data
    """
    * @brief Function to return the length of the dataset.
    * @param None.
    * @return The length of the dataset.
    """
    def __len__(self):
        return len(self.csv_data)
    """
    * @brief Function to return one data point from the csv file.
    * @param The index of the data to return.
    * @return The data value of the index
    """ 
    def __getitem__(self,i):
        data_entry = self.csv_data[i]
        # Splitting the data of the features from one single data point.
        to_return = {
            'img_center_pth': data_entry[0],
            'img_left_pth': data_entry[1],
            'img_right_pth': data_entry[2],
            'steering_angle': data_entry[3],
            'throttle': data_entry[4],
            'brake': data_entry[5],
            'speed': data_entry[6]
        }

        return to_return

# Main function to test the Features class #
# if __name__ == '__main__':
#     dataset = Features()
#     print(dataset[0]['img'].shape)
