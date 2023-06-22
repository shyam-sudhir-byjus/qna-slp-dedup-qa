from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name, from_pt=True)

def compare_sentences_using_bert(sentence1, sentence2):
    encoded_input = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors="tf")
    
    outputs = model(encoded_input.input_ids)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    
    similarity_score = np.inner(embeddings[0], embeddings[1])
    return similarity_score


if __name__ == "__main__":
    pass
    # df = pd.read_csv("blueprint_similar_question.csv")[['topic','SUBTOPIC_NAME']]
    # blue_subtopics = df['topic'].tolist()
    # qna_subtopics = df['SUBTOPIC_NAME'].tolist()
    # for i in range(len(blue_subtopics)):
    #     sim_score = compare_sentences_using_bert(blue_subtopics[i], qna_subtopics[i])
    #     print(f'{blue_subtopics[i]} - {qna_subtopics[i]} - {sim_score}')

    

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# def preprocess_text(text):
#     tokens = nltk.word_tokenize(text)
#     tokens = [token.lower() for token in tokens if token.lower() not in
#               set(stopwords.words('english'))]
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     return tokens


# def compare_sentences_nltk(sentence1, sentence2):
#     processed1 = preprocess_text(sentence1)
#     processed2 = preprocess_text(sentence2)

#     intersection = len(set(processed1).intersection(set(processed2)))
#     union = len(set(processed1)) + len(set(processed2)) - intersection

#     jacc_sim = intersection/union
#     return jacc_sim
    
# s = time.time()
# print(compare_sentences_using_bert('Atomic Valency', 'Valency and Stability'))
# print(time.time()-s)
# print(compare_sentences_nltk('Atomic Valency', 'Valency and Stability'))

# print(compare_sentences_using_bert('Types of Chemical Reaction', 'Introduction to Chemical Changes'))
# print(compare_sentences_nltk('Types of Chemical Reaction', 'Introduction to Chemical Changes'))