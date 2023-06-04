class DataEnhancer:
    def enhance(self, data_frame):
        data_frame['Tags Count'] = data_frame['Tags'].apply(lambda x: len(x))
        return data_frame
