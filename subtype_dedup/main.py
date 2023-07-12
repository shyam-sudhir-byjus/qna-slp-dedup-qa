from string_matching import boyer_moore, kmp_algorithm
from preprocess import preprocess_text
from word_embedding import sbert_similarity
import json
import warnings
warnings.filterwarnings('ignore')

with open('updated_questions.json') as file:
    updated_data = json.load(file)

questions = [question_set['question'] for question_set in updated_data['questions']]
subquestions = [question_set['subquestion'] for question_set in updated_data['questions']]

if __name__ == "__main__":
    for i in range(len(questions))[:5]:
        text1, text2 = questions[i], subquestions[i]
        
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)
        text1 = ' '.join(text1)
        text2 = ' '.join(text2)
        print(text1)
        print(text2)
        
        index_of_start = (kmp_algorithm(text1, text2))[1]
        if index_of_start != -1:
            sim_score = sbert_similarity(text1[index_of_start:], text2)
        
        print("-"*50)
        
    