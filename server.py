from flask import Flask, request, jsonify
from qna_blueprint_tags import find_tags_for_similarity
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/get_duplicates', methods=['POST'])
def process_questions():
    data = request.get_json()
    question_A = data["question_1"]
    question_B = data["question_2"]

    response = find_tags_for_similarity(question_A, question_B)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8000,debug=True)