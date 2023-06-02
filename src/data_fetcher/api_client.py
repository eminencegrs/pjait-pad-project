from params import Params
import requests
import sys

class ApiClient:
    def __init__(self, config):
        self.api_key = config['stack_overflow'].get('api_key', None)
        self.api_url = config['stack_overflow'].get('api_url', None)

    def get_resource(self, resource, pagesize = 100, page = 1, tagged = None):
        request_url = self.api_url + '/' + resource
        params = Params.create(self.api_key, pagesize, page)
        
        if tagged:
            params['tagged'] = tagged

        response = requests.get(request_url, params = params)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data:
                return data
            else:
                print("Error: Invalid data received")
                return None
        else:
            print("Error:", response.status_code)
            return None

    def get_paginated_questions(self, pagesize = 10, max_pages = 10, tagged = None):
        page = 1
        all_questions = []

        print(f"Fetching questions from StackOverflow (Tag: {tagged}). Page size: {pagesize}. Max pages: {max_pages}.")

        while True:
            message = f"Fetching page {page}..."
            print(message, end="", flush=True)
            data = self.get_resource("questions", pagesize = pagesize, page = page, tagged = tagged)
            if data:
                questions = data['items']
                all_questions.extend(questions)

            if not data or not data['has_more'] or (max_pages is not None and page >= max_pages):
                break

            page += 1
            sys.stdout.write("\r\033[K")

        print(f"\r{len(all_questions)} questions have been received (for '{tagged}').\r")
        
        return all_questions
    
    def get_comments(self, pagesize=100, page=1):
        request_url = self.api_url + '/comments'
        params = Params.create(self.api_key, pagesize, page)

        response = requests.get(request_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data:
                return data
            else:
                print("Error: Invalid data received")
                return None
        else:
            print("Error:", response.status_code)
            return None

    def get_paginated_comments(self, pagesize=100, max_pages=100):
        page = 1
        all_comments = []

        print(f"Fetching comments from StackOverflow. Page size: {pagesize}. Max pages: {max_pages}.")

        while True:
            message = f"Fetching page {page}..."
            print(message, end="", flush=True)
            data = self.get_comments(pagesize=pagesize, page=page)
            if data:
                questions = data['items']
                all_comments.extend(questions)

            if not data or not data['has_more'] or (max_pages is not None and page >= max_pages):
                break

            page += 1
            sys.stdout.write("\r\033[K")

        print(f"\r{len(all_comments)} comments have been received.\r")
        
        return all_comments

    def get_answers(self, pagesize=100, page=1):
        request_url = self.api_url + '/answers'
        params = Params.create(self.api_key, pagesize, page)

        response = requests.get(request_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data:
                return data
            else:
                print("Error: Invalid data received")
                return None
        else:
            print("Error:", response.status_code)
            return None

    def get_paginated_answers(self, pagesize=100, max_pages=100):
        page = 1
        all_comments = []

        print(f"Fetching answers from StackOverflow. Page size: {pagesize}. Max pages: {max_pages}.")

        while True:
            message = f"Fetching page {page}..."
            print(message, end="", flush=True)
            data = self.get_comments(pagesize=pagesize, page=page)
            if data:
                questions = data['items']
                all_comments.extend(questions)

            if not data or not data['has_more'] or (max_pages is not None and page >= max_pages):
                break

            page += 1
            sys.stdout.write("\r\033[K")

        print(f"\r{len(all_comments)} answers have been received.\r")
        
        return all_comments
