

import nltk
import json
import string 
import re
import en_core_web_sm
import spacy
import emoji
import torch 

from transformers import AutoModelWithLMHead, AutoTokenizer

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS


MAX_LEN = 512
SUMMARY_LEN = 150

nlp = en_core_web_sm.load()
stopwords = list(STOP_WORDS)

punctuations = string.punctuation
wordnet_lemmatizer = WordNetLemmatizer()

model1 = AutoModelWithLMHead.from_pretrained("sshleifer/distilbart-cnn-12-6", )
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
path = "./Trained_Models/summarization_bart_model.pt"
model1.load_state_dict(torch.load(path))
    

def clean_text(sent):
    sentence = sent.strip()
#     sentence = sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    tokens = [token for token in tokens if not (token in punctuations or token in stopwords)]
    
    lemmatized_token = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    new_sentence = " ".join(lemmatized_token)
    return new_sentence

### Web scraping

def scrape_web_data(link):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}

#     links = df_links["Link"]

    tags = ["span","title","time","p", "h4"]
    c = 0
    summ_li = []
#     for link in links:
    source = Request(url = link, headers = headers)
    html = urlopen(source).read()
    soup = BeautifulSoup(html)
    para = ""
    for elem in soup():    
        if((elem.name in tags)):
            text = elem.text
            para += text +" "
    
    para = para.strip()
    para = para.replace("\n", " ")
    para = re.sub(r'<.*?>', '', para)
    para = emoji.get_emoji_regexp().sub(r"", para)  
    
    return para

### function definition to predict the summaries
def predict_summaries(text):
    article_input_ids = tokenizer.batch_encode_plus([text], max_length= MAX_LEN, pad_to_max_length=True,return_tensors='pt')
    summary_ids = model1.generate(
                input_ids = article_input_ids['input_ids'], 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )

    summary_txt = [tokenizer.decode(g , skip_special_tokens=True) for g in summary_ids]
    return (" ".join(summary_txt))


