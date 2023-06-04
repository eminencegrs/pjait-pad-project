class DataCleaner:
    COLUMNS_TO_CLEAR = ['accepted_answer_id', 'last_edit_date', 'closed_date', 'closed_reason', 'owner.reputation']
    
    def clean_data(self, data_frame):
        for column in self.COLUMNS_TO_CLEAR:
            if column == 'owner.reputation':
                data_frame[column] = data_frame[column].fillna(0)
            else:
                data_frame[column] = data_frame[column].fillna('N/A')
        return data_frame
