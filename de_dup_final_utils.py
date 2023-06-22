import itertools
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from datasketch import MinHash, MinHashLSH
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import string
from pylatexenc.latex2text import LatexNodes2Text
import re
import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


lemmatizer = WordNetLemmatizer()
MATHS_LIST = ['Mathematics', 'Discrete Mathematics', 'Engineering Mathematics']
num_perm = 256


def math_quest_process(_text):
    _text = str(_text)
    _text = _text.replace("<li>", " <li>")
    _text = BeautifulSoup(_text).text
    char_present = list(filter(lambda x: x in _text, "{}[]()-=+/*,"))
    _text = list(map(lambda x: x.split("\\"), _text.split()))
    _text = list(itertools.chain.from_iterable(_text))
    for char in char_present:
        _text = list(map(lambda x: re.split(f"([{char}])", x), _text))
        _text = list(itertools.chain.from_iterable(_text))
    _text = list(map(lambda x: x.strip(), _text))
    _text = list(filter(lambda x: x != "", _text))
    return list(_text)


def non_math_quest_process(clean_question):
    clean_question = BeautifulSoup(clean_question).get_text(separator=u' ')
    clean_question = re.sub(r'[^\w\s_]', ' ', clean_question)
    # clean_question = clean_question.replace("_", "")
    que_token = word_tokenize(clean_question)
    que_token = list(map(lambda x: x.lower(), que_token))
    # py_nltk2 = [t for t in que_token if t not in stopwords.words ('english')]
    return que_token  # py_nltk2


def text_split(_text, subject):
    if subject in MATHS_LIST:
        word_list = math_quest_process(_text)
    else:
        word_list = non_math_quest_process(_text)
    word_list = list(map(lambda word: str(word).lower(), word_list))
    return set(word_list)


def min_hash_set(_set):
    m = MinHash(num_perm=num_perm)
    for shingle in _set:
        m.update(shingle.encode('utf8'))
    return m


def jac_sim(_dict, c1, c2):
    s1 = set(_dict[c1])
    s2 = set(_dict[c2])
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    if union:
        return inter / union


def s1_intersect(_dict, c1, c2):
    s1 = set(_dict[c1])
    s2 = set(_dict[c2])
    intersection = s1.intersection(s2)
    return s1 - intersection


def s2_intersect(_dict, c1, c2):
    s1 = set(_dict[c1])
    s2 = set(_dict[c2])
    intersection = s1.intersection(s2)
    return s2 - intersection

def id_sort(_dict):
    return sorted([int(_dict["ID1"]), int(_dict["ID2"])])

def remove_latex(txt):
    txt = txt.replace("\r\n", ' ')
    txt = txt.replace("\n", '')
    txt = txt.replace("\xa0"," ")
    txt = txt.replace("}{","}/{")
    txt = txt.replace("\\sqrt","‚àö")
    whether_latex = False
    if "$$" in txt:
        txt = txt.replace(" $ ","")
        txt = txt.replace("$$","$")
        txt = re.sub(r'[$]', " $ ", txt)
        pattern = re.compile(r"\$ (.*?) \$", re.DOTALL)
        raw_matches = re.findall(pattern, txt)
        raw_matches = list(set(raw_matches))
        raw_matches = [match.lstrip() for match in raw_matches]
        raw_matches = [match.rstrip() for match in raw_matches]
        matches = list(set(raw_matches))
        for l in matches:
            whether_latex = True
            cls = LatexNodes2Text().latex_to_text(l)
            cls = cls.replace(" ","")
            txt = txt.replace(l, cls)
        txt = txt.replace("$","")
    else:
        pattern = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
        matches = re.findall(pattern, txt)
        matches = list(set(matches))
        for l in matches:
            whether_latex = True
            cls = LatexNodes2Text().latex_to_text(l)
            cls = cls.replace(" ","")
            txt = txt.replace(l, cls)
    txt = txt.lstrip()
    txt = txt.rstrip()
    txt = re.sub(r'[\^]', "", txt)
    txt = re.sub(r'[_]', "", txt)
    txt = re.sub(r'[{]', "", txt)
    txt = re.sub(r'[}]', "", txt)
    txt = re.sub(r'[(]', "", txt)
    txt = re.sub(r'[)]', "", txt)
    txt = re.sub(r'[\\]', "", txt)
    # txt = re.sub(r'[/]', "", txt)
    txt = txt.replace("  "," ")
    return txt, whether_latex


def clean_text(text):
    clean_question = BeautifulSoup(text, features='lxml').get_text(separator=u' ')
    clean_question, _ = remove_latex(clean_question)
    return clean_question


def preprocess_text_2(clean_question):
    clean_question = BeautifulSoup(clean_question).get_text(separator=u' ')
    clean_question, _ = remove_latex(clean_question)
    que_token = tokenize.wordpunct_tokenize(clean_question)
    que_token = list(map(lambda x: x.lower(), que_token))
    que_token = [t for t in que_token if t not in string.punctuation]
    que_token = [t for t in que_token if t not in stopwords.words('english')]
    que_token = [lemmatizer.lemmatize(t) for t in que_token]
    return que_token


def text_split_2(_text):
    word_list = preprocess_text_2(_text)
    return word_list


def remove_digits_formulae(_text):
    clean_question = BeautifulSoup(_text, features='lxml').get_text(separator=u' ')
    clean_question, _ = remove_latex(clean_question)
    clean_question = re.sub(r"\$.*?\$", "", clean_question)
    # clean_question = re.sub(r"\b\d+\b|\b\w{1,2}\b","",clean_question)
    clean_question = re.sub(r"\d+","", clean_question)
    que_token = tokenize.wordpunct_tokenize(clean_question)
    que_token = list(map(lambda x: x.lower(), que_token))
    que_token = [t for t in que_token if t not in (string.punctuation + '/')]
    que_token = [t for t in que_token if t not in stopwords.words('english')]
    que_token = [lemmatizer.lemmatize(t) for t in que_token]
    return que_token


def preprocess_text_6(clean_question):
    clean_question = BeautifulSoup(clean_question, features='lxml').get_text(separator=u' ')
    clean_question = re.sub(r'[^\w\s_]', ' ', clean_question)
    que_token = tokenize.word_tokenize(clean_question)
    que_token = list(map(lambda x: x.lower(), que_token))
    que_token = [lemmatizer.lemmatize(t) for t in que_token]
    return que_token


def text_split_6(_text):
    word_list = preprocess_text_6(_text)
    return word_list


def preprocess_text(clean_question):
    clean_question = BeautifulSoup(clean_question).get_text(separator=u' ')
    clean_question = re.sub(r'[^\w\s_]', ' ', clean_question)
    #     clean_question = clean_question.replace("_", "")
    # print(clean_question)
    que_token = tokenize.wordpunct_tokenize(clean_question)
    que_token = list(map(lambda x: x.lower(), que_token))
    que_token = [t for t in que_token if t not in string.punctuation]
    que_token = [t for t in que_token if t not in stopwords.words('english')]
    # que_token = [lemmatizer.lemmatize(t) for t in que_token]
    return que_token


def text_split2(_text):
    word_list = preprocess_text(_text)
    return word_list


def get_images(b):
    return [img['src'] for img in b.findAll('img')]


def get_semantic_similarity(word1, word2):
    model = SentenceTransformer('bert-base-nli-mean-tokens')  

    word1_embedding = model.encode(word1, convert_to_tensor=True).reshape(-1,1)
    word2_embedding = model.encode(word2, convert_to_tensor=True).reshape(-1,1)

    similarity = cosine_similarity(word1_embedding, word2_embedding)[0][0]
    return similarity


def get_noun31_noun32_similarity(nouns1, nouns2):
    sim_list = []
    for noun1 in nouns1:
        for noun2 in nouns2:
            if noun1 == noun2:
                continue
            sim = get_semantic_similarity(noun1, noun2)
            if (noun1, noun2 ,sim) not in sim_list and sim > 0.5:
                sim_list.append((noun1, noun2 ,sim))
    
    return len(sim_list) >= 1 and len(sim_list) <= 4


def check_noun_similarity(nouns1, nouns2):
    nouns1 = [t for t in tokenize.wordpunct_tokenize(" ".join(math_quest_process(' '.join(nouns1))).replace("_", "")) if t not in string.punctuation]
    nouns2 = [t for t in tokenize.wordpunct_tokenize(" ".join(math_quest_process(' '.join(nouns2))).replace("_", "")) if t not in string.punctuation]
    
    nouns1 = list(filter(lambda x: len(x) >= 3, nouns1))
    nouns2 = list(filter(lambda x: len(x) >= 3, nouns2))
    if len(nouns1) < 4 and len(nouns2) < 4:
        return
    
    nouns1_temp = nouns1.copy()
    nouns2_temp = nouns2.copy()

    nouns1 = [lemmatizer.lemmatize(noun.lower().encode('ascii','ignore').decode('ascii')) for noun in nouns1_temp if noun not in nouns2_temp]
    nouns2 = [lemmatizer.lemmatize(noun.lower().encode('ascii','ignore').decode('ascii')) for noun in nouns2_temp if noun not in nouns1_temp]
    
    pos_tags_nouns1 = pos_tag(nouns1)
    pos_tags_nouns2 = pos_tag(nouns2)

    nouns1 = [word for word, pos in pos_tags_nouns1 if pos not in ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
    nouns2 = [word for word, pos in pos_tags_nouns2 if pos not in ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
    
    return get_noun31_noun32_similarity(nouns1, nouns2)    


def get_preds(row):
    s1 = row.clean_question1
    s2 = row.clean_question2
    f1 = row.f1
    f2 = row.f2
    f3 = row.f3
    f_num = row.f_num
    f_noun = row.f_noun

    if math.isnan(f_num):
        f_num = 0
    if math.isnan(f_noun):
        f_noun = 0
    if math.isnan(f2):
        f2 = 0
    
    dup_list = []

    if f1 >= 0.95:
        dup_list.append(['exact'])
    
    b1 = BeautifulSoup(s1, features='lxml')
    b2 = BeautifulSoup(s2, features='lxml')
    b1t = b1.get_text(separator=u' ')
    b2t = b2.get_text(separator=u' ')
    im1 = get_images(b1)
    im2 = get_images(b2)
    tokens1 = preprocess_text(b1t)
    tokens2 = preprocess_text(b2t)
    if len(tokens1) <= 4 or len(tokens2) <= 4 or len(im1) > 0 or len(im2) > 0:
        return 0, [""]
    
    if f3 >= 0.8 and check_noun_similarity(row.nn_31, row.nn_32):
        dup_list.append(["noun_similarity"])
        
    if f_num <= 0.4 and f2 >= 0.8:
       dup_list.append(["value_similarity"])
        

    if len(dup_list) > 1:
        return 1, list(itertools.chain.from_iterable(dup_list))
    elif len(dup_list) == 1:
        return 1, dup_list[0]

    return 0, [""]



def math_quest_process(_text):
    regex_list = [
        r'\s*\((?:\d+|[a-zA-Z])\s*Marks?\)|\s*\[(?:\d+|[a-zA-Z])\s*Marks?\]',
        r'\s*\(\d+\s*Marks?\)',
        r'(\(|\[)\s*\d+\s*Marks?\s*(\)|\])?$',
        r'(\(|\[)\d+ Marks?(\)|\])$',
        r'\s*[\(\[]\d+\s*Marks?[\)\]]$',
        r'(\[|\()\s*\d+\s*Marks?\s*(\]|\))?$',
        r'(\[|\()\s*\d+\s*marks?\s*(\]|\))?$',
        r'^(Question|question)\s+[1-9][0-9]*(\(\w+\))?\s+',
        r'^(Question|question)\s+([1-9][0-9]*|[ivxcl]+(\s*\(.*\))?)\s+'
        r'^(Questions|questions)\s+[1-9][0-9]*(\(\w+\))?\s+',
        r'^(Questions|questions)\s+([1-9][0-9]*|[ivxcl]+(\s*\(.*\))?)\s+'
    ]
    _text = str(_text)
    _text = _text.replace("<li>", " <li>")
    _text = BeautifulSoup(_text).text
    _text = re.sub('\n', ' ', _text)
    _text = _text.strip()
    for _r in regex_list:
        _text = re.sub(_r, '', _text)
        _text = _text.strip()

    char_present = list(filter(lambda x: x in _text, "{}[]()-=+/*,"))
    _text = list(map(lambda x: x.split("\\"), _text.split()))
    _text = list(itertools.chain.from_iterable(_text))
    for char in char_present:
        _text = list(map(lambda x: re.split(f"([{char}])", x), _text))
        _text = list(itertools.chain.from_iterable(_text))
    _text = list(map(lambda x: x.strip(), _text))
    _text = list(filter(lambda x: x != "", _text))
    return list(_text)



def extract_numbers(_string):
    _string = " ".join(math_quest_process(_string))
    pattern = r"-?\d+(?:\.\d+)?"
    numbers = re.findall(pattern, _string)
    return [float(num) for num in numbers]


def remove_digits_formulae(_text):
    clean_question = BeautifulSoup(_text, features='lxml').get_text(separator=u' ')
    clean_question, _ = remove_latex(clean_question)
    clean_question = re.sub(r"\$.*?\$", "", clean_question)
    clean_question = re.sub(r"\b\d+\b|\b\w{1,2}\b","",clean_question)
    clean_question = re.sub(r"\d+","", clean_question)
    que_token = tokenize.wordpunct_tokenize(clean_question)
    que_token = list(map(lambda x: x.lower(), que_token))
    que_token = [t for t in que_token if t not in (string.punctuation + '/|')]
    que_token = [t for t in que_token if t not in stopwords.words('english')]
    for t in que_token:
        if all(s in (string.punctuation + '/|') for s in t):
            que_token.remove(t)
    que_token = [lemmatizer.lemmatize(t) for t in que_token]
    return que_token


def find_max_substring(sentence1, sentence2):
    m = len(sentence1)
    n = len(sentence2)

    table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if sentence1[i - 1] == sentence2[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    substring = ""
    i = m
    j = n
    while i > 0 and j > 0:
        if sentence1[i - 1] == sentence2[j - 1]:
            substring = sentence1[i - 1] + substring
            i -= 1
            j -= 1
        elif table[i - 1][j] > table[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return substring


def kmp_algorithm(_dict, text, pattern):
    qna_ques, blue_ques = _dict[text], _dict[pattern]
    max_substring = len(find_max_substring(qna_ques, blue_ques))
    max_substring1 = len(find_max_substring(blue_ques, qna_ques))
    len_ques = min(len(qna_ques), len(blue_ques))
    avg1 = (abs(max_substring-len_ques)/(max_substring+len_ques)) * 100
    avg2 = (abs(max_substring1-len_ques)/(max_substring1+len_ques)) * 100

    if avg1 < 10 or avg2 < 10:
        return True
    return False


def noun_detection(_text):
    clean_question = BeautifulSoup(_text, features='lxml').get_text(separator=u' ')
    clean_question, _ = remove_latex(clean_question)
    words = word_tokenize(clean_question)
    tagged_words = pos_tag(words)
    nouns = [word for word, pos in tagged_words if pos.startswith("N")]
    return nouns


def remove_nouns(_text):
    nouns = noun_detection(_text)
    clean_question = BeautifulSoup(_text, features='lxml').get_text(separator=u' ')
    clean_question, _ = remove_latex(clean_question)
    words = word_tokenize(clean_question)
    words_without_nouns = [word for word in words if word not in nouns]
    return words_without_nouns
