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


def kmp_algorithm(text, pattern):
    qna_ques, blue_ques = text, pattern
    max_1 = find_max_substring(qna_ques, blue_ques).strip()
    max_2 = find_max_substring(blue_ques, qna_ques).strip()
    try:
        index_1 = qna_ques.index(max_1)
    except:
        index_1 = -1
    try:
        index_2 = blue_ques.index(max_2)
    except:
        index_2 = -1
    
    index_of_substring = max(index_1, index_2)
    max_substring = len(max_1)
    max_substring1 = len(max_2)
    len_ques = min(len(qna_ques), len(blue_ques))
    avg1 = (abs(max_substring-len_ques)/(max_substring+len_ques)) * 100
    avg2 = (abs(max_substring1-len_ques)/(max_substring1+len_ques)) * 100

    if avg1 < 5 or avg2 < 5:
        return True, index_of_substring
    return False, -1


"""
def boyer_moore(text, pattern):
    m = len(pattern)
    n = len(text)

    skip = [m] * 256
    for i in range(m - 1):
        skip[ord(pattern[i])] = m - i - 1

    i, j = m - 1, m - 1  

    while j < n:
        if pattern[i] == text[j]:
            if i == 0:
                return j  
            else:
                i -= 1
                j -= 1
        else:
            j += skip[ord(text[j])]
            i = m - 1

    return -1 
"""