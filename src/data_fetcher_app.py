import sys
import os
import pandas as pd
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'data_fetcher'))

from api_client import ApiClient
from config_reader import ConfigReader
from csv_file_writer import CsvFileWriter
from dataframe_factory import DataFrameFactory
from meta import Meta

if __name__ == "__main__":
    config_path = os.path.join(current_dir, 'settings', 'appconfig.json')
    config = ConfigReader(config_path).read_config()
    api_client = ApiClient(config)
    data_frame_factory = DataFrameFactory()
    csv_file_writer = CsvFileWriter()

    languages = config['languages_to_analyze']
    questions_list = []

    for language in languages:
        print(f"Getting questions for '{language}'...")
        questions = api_client.get_paginated_questions(pagesize = 100, max_pages = 100, tagged = language)
        questions_list.extend(questions)
        time.sleep(1)

    unique_questions = {q['question_id']: q for q in questions_list}.values()

    questions_df = data_frame_factory.create_normalized_dataframe(
        unique_questions,
        meta = Meta.get_question_meta())

    questions_df['last_activity_date'] = pd.to_datetime(questions_df['last_activity_date'], unit='s')
    questions_df['creation_date'] = pd.to_datetime(questions_df['creation_date'], unit='s')
    questions_df['closed_date'] = pd.to_datetime(questions_df['closed_date'], unit='s')
    questions_df['last_edit_date'] = pd.to_datetime(questions_df['last_edit_date'], unit='s')
    questions_df = questions_df.drop(columns = ['content_license', 'owner.user_type', 'owner.profile_image', 'owner.link'])
    
    def find_common_tag(tags):
        for tag in tags:
            if tag in languages:
                return tag
        return None
    
    questions_df['language'] = questions_df['tags'].apply(find_common_tag)

    print(questions_df)
    print(questions_df.describe())
    csv_file_writer.write(questions_df, "questions")
