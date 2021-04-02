import requests
import sys
from config.config import POST_DATA_URL

def send_data(data):
    requests.post(POST_DATA_URL, json=data)