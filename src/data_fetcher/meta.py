class Meta:
    def get_question_meta():
        return [
            'tags',
            'is_answered',
            'view_count',
            'answer_count',
            'score',
            'last_activity_date',
            'creation_date',
            'question_id',
            'content_license',
            'link',
            'title',
            ['owner', 'account_id'],
            ['owner', 'reputation'],
            ['owner', 'user_id'],
            ['owner', 'user_type'],
            ['owner', 'display_name'],
            ['owner', 'link'],
            ['owner', 'accept_rate'],
            'last_edit_date',
            'closed_date',
            'closed_reason',
            'accepted_answer_id',
        ]
        
    def get_comment_meta():
        return [
            'edited',
            'score',
            'creation_date',
            'post_id',
            'comment_id',
            'content_license',
            ['owner', 'account_id'],
            ['owner', 'reputation'],
            ['owner', 'user_id']
        ]
