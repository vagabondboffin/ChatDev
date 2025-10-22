
import requests
from bs4 import BeautifulSoup
import openai
from openai import OpenAI
import wikipediaapi
import os
import time

# Modified OpenAI client initialization that works with Ollama
self_api_key = os.environ.get('OPENAI_API_KEY', 'ollama')  # Default to 'ollama'
BASE_URL = os.environ.get('BASE_URL')

try:
    if BASE_URL:
        # For Ollama compatibility, use a simpler initialization
        client = OpenAI(
            api_key=self_api_key,
            base_url=BASE_URL,
        )
    else:
        client = OpenAI(api_key=self_api_key)
except TypeError as e:
    # Fallback if there's an initialization error (like proxies issue)
    print(f"OpenAI client initialization warning: {e}")
    print("Using fallback client configuration...")
    client = OpenAI(api_key=self_api_key, base_url=BASE_URL) if BASE_URL else OpenAI(api_key=self_api_key)

    
def get_baidu_baike_content(keyword):
    # design api by the baidubaike
    url = f'https://baike.baidu.com/item/{keyword}'
    # post request
    response = requests.get(url)

    # Beautiful Soup part for the html content
    soup = BeautifulSoup(response.content, 'html.parser')
    # find the main content in the page
    # main_content = soup.find('div', class_='lemma-summary')
    main_content = soup.contents[-1].contents[0].contents[4].attrs['content']
    # find the target content
    # content_text = main_content.get_text().strip()
    return main_content


def get_wiki_content(keyword):
    #  Wikipedia API ready
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
    #the topic content which you want to spider
    search_topic = keyword
    # get the page content
    page_py = wiki_wiki.page(search_topic)
    # check the existence of the content in the page
    if page_py.exists():
        print("Page - Title:", page_py.title)
        print("Page - Summary:", page_py.summary)
    else:
        print("Page not found.")
    return page_py.summary



def modal_trans(task_dsp):
    try:
        task_in ="'" + task_dsp + \
               "'Just give me the most important keyword about this sentence without explaining it and your answer should be only one keyword."
        messages = [{"role": "user", "content": task_in}]
        response = client.chat.completions.create(messages=messages,
        model="gpt-3.5-turbo-16k",
        temperature=0.2,
        top_p=1.0,
        n=1,
        stream=False,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={})
        response_text = response.choices[0].message.content
        spider_content = get_wiki_content(response_text)
        # time.sleep(1)
        task_in = "'" + spider_content + \
               "',Summarize this paragraph and return the key information."
        messages = [{"role": "user", "content": task_in}]
        response = client.chat.completions.create(messages=messages,
        model="gpt-3.5-turbo-16k",
        temperature=0.2,
        top_p=1.0,
        n=1,
        stream=False,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={})
        result = response.choices[0].message.content
        print("web spider content:", result)
    except:
        result = ''
        print("the content is none")
    return result