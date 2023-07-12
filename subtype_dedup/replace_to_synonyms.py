import nltk
from nltk.corpus import wordnet
import json

nltk.download('wordnet')

def replace_with_synonyms(text):
    tokens = nltk.word_tokenize(text)
    synonyms = []

    for token in tokens:
        synsets = wordnet.synsets(token)
        if synsets:
            synonyms.append(synsets[0].lemmas()[0].name())
        else:
            synonyms.append(token)

    return ' '.join(synonyms)

with open('original_questions.json') as file:
    original_data = json.load(file)

updated_data = {"questions": []}

for question_set in original_data["questions"]:
    updated_question = question_set["question"]
    updated_subquestion = replace_with_synonyms(question_set["subquestion"])
    updated_question_set = {
        "question": updated_question,
        "subquestion": updated_subquestion
    }
    updated_data["questions"].append(updated_question_set)

with open('updated_questions.json', 'w') as file:
    json.dump(updated_data, file, indent=4)
