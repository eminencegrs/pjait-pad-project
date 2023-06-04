import sys
import os
import pandas as pd

class DataReader:
    DATASET_DIR_NAME = 'dataset'
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.current_dir)
        sys.path.insert(0, os.path.join(self.parent_dir, self.DATASET_DIR_NAME))
        
    def read_data(self):
        dataset_path = os.path.join(self.parent_dir, self.DATASET_DIR_NAME, self.csv_file)
        data_frame = pd.read_csv(dataset_path, low_memory=False)
        return data_frame
