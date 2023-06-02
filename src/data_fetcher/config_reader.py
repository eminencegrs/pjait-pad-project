import json

class ConfigReader:
    def __init__(self, config_file_name):
        self.config_file_name = config_file_name
        
    def read_config(self):
        with open(self.config_file_name, 'r') as f:
            config = json.load(f)
        return config
