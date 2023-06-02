class Params:
    def create(api_key, page_size, page):
        return {
            'pagesize': page_size,
            'order': 'desc',
            'sort': 'creation',
            'site': 'stackoverflow',
            'page': page,
            'key': api_key
        }
