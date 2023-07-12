from sentence_transformers import SentenceTransformer, util

def sbert_similarity(words1, words2):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences1 = words1.split(' ')
    sentences2 = words2.split(' ')
    
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores

"""

from transformers import AutoTokenizer, AutoModel
from preprocess import preprocess_text as preprocess_question
tokenizer = AutoTokenizer.from_pretrained("openai/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("openai/all-MiniLM-L6-v2")


def calculate_similarity(smaller_question, larger_question):
    preprocessed_larger_question = preprocess_question(larger_question)
    preprocessed_smaller_question = preprocess_question(smaller_question)

    smaller_question_encoded = tokenizer.encode(preprocessed_smaller_question, truncation=True, padding=True, return_tensors="pt")
    larger_question_encoded = tokenizer.encode(preprocessed_larger_question, truncation=True, padding=True, return_tensors="pt")

    smaller_question_embedding = model(smaller_question_encoded.input_ids)[0].squeeze(0)
    larger_question_embedding = model(larger_question_encoded.input_ids)[0].squeeze(0)

    similarity_score = smaller_question_embedding.cosine_similarity(larger_question_embedding).item()
    return similarity_score
    
"""

