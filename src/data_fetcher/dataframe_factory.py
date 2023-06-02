import pandas as pd

class DataFrameFactory:
    def create_dataframe(self, items):
        data = [{
            'question_id': q['question_id'],
            'title': q['title'],
            'link': q['link'],
            'creation_date': pd.to_datetime(q['creation_date'], unit='s'),
            'score': q['score'],
            'view_count': q['view_count'],
            'answer_count': q['answer_count'],
            'tags': ', '.join(q['tags'])
        } for q in items]

        df = pd.DataFrame(data)
        return df
    
    def create_normalized_dataframe(self, items, meta):
        data = pd.json_normalize(items, meta = meta)
        return data
