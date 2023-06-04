class ColumnFilter:
    DATASET_COLUMNS = [
        'question_id',
        'title',
        'language',
        'score',
        'is_answered',
        'accepted_answer_id',
        'view_count',
        'answer_count',
        'creation_date',
        'last_edit_date',
        'last_activity_date',
        'closed_date',
        'closed_reason',
        'owner.account_id',
        'owner.reputation',
        'tags'
    ]
    
    def filter_data(self, data_frame):
        return data_frame[self.DATASET_COLUMNS]
