from datetime import datetime
import os

class CsvFileWriter:
    def __init__(self, directory_name=None):
        self.directory_name = "results" if directory_name is None else directory_name
        
    def write(self, data_frame, name_prefix):
        if not os.path.isdir(self.directory_name):
            os.makedirs(self.directory_name)
        file_name = f"{name_prefix}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
        data_frame.to_csv(f"{self.directory_name}/{file_name}", index=False)
