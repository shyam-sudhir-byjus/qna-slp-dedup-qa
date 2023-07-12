import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk import tokenize
import string
from bs4 import BeautifulSoup
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')

def get_root_word(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    return lemma

def replace_synonyms_with_root_word(sentence):
    tokens = word_tokenize(sentence)
    replaced_tokens = []

    for token in tokens:
        root_word = get_root_word(token)
        synonyms = wordnet.synsets(token)

        if synonyms:
            synonyms = [syn.lemmas()[0].name() for syn in synonyms]
            if root_word not in synonyms:
                replaced_tokens.append(root_word)
            else:
                replaced_tokens.append(token)
        else:
            replaced_tokens.append(root_word)

    replaced_sentence = ' '.join(replaced_tokens)
    return replaced_sentence


def preprocess_text_initial(clean_question):
    clean_question = BeautifulSoup(clean_question).get_text(separator=u' ')
    clean_question = re.sub(r'[^\w\s_]', ' ', clean_question)
    que_token = tokenize.wordpunct_tokenize(clean_question)
    que_token = list(map(lambda x: x.lower(), que_token))
    que_token = [t for t in que_token if t not in string.punctuation]
    que_token = [t for t in que_token if t not in stopwords.words('english')]
    return que_token


def preprocess_text(text):
    text = replace_synonyms_with_root_word(text)
    tokens = preprocess_text_initial(text)
    
    pos_tags = pos_tag(tokens)
    # print(pos_tags)
    tokens = [token for token, pos in pos_tags if pos not in ['CD']]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # preprocessed_text = ' '.join(tokens)
    return tokens
