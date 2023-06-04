class ColumnRenamer:
    COLUMN_MAPPING = {
        'question_id':          'ID',
        'title':                'Title',
        'language':             'Language',
        'score':                'Score',
        'is_answered':          'Is Answered?',
        'accepted_answer_id':   'Answer ID',
        'view_count':           'View Count',
        'answer_count':         'Answer Count',
        'creation_date':        'Creation Date',
        'last_edit_date':       'Edit Date',
        'last_activity_date':   'Last Activity Date',
        'closed_date':          'Closed Date',
        'closed_reason':        'Closed Reason',
        'owner.account_id':     'Owner ID',
        'owner.reputation':     'Owner Reputation',
        'tags':                 'Tags'
    }

    def rename_columns(self, data_frame):
        data_frame.rename(columns=self.COLUMN_MAPPING, inplace=True)
        return data_frame
