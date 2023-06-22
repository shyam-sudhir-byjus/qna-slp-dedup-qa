import csv
import os
import sys
import warnings
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer, WordNetLemmatizer

from datasketch import MinHashLSH
from tqdm import tqdm
from datetime import datetime
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from da_lpp.de_dup_final_utils import s1_intersect, s2_intersect, jac_sim, \
    text_split_2, text_split_6, text_split, min_hash_set, get_preds, remove_digits_formulae, \
        extract_numbers, remove_nouns, noun_detection
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

import pandas as pd
import html2text


def make_pkl_file(sample_questions, id_list):
    sample_questions = [html2text.html2text(q).replace("\n","") for q in sample_questions]
    questions  = sample_questions 
    ids = [i for i in range(len(questions))]#pickled_data["ID"].tolist()
    urls =["temp"]*len(questions) #pickled_data["url"].tolist()
    subjects = urls
    sources = urls
    ques_ids = id_list

    df = pd.DataFrame(list(zip(ids, urls, questions, questions, questions, questions, sources, subjects, ques_ids)), 
                        columns=[
                                "ID",
                                "url",
                                "raw_title",
                                "clean_question",
                                "raw_solution",
                                "clean_solution",
                                "SOURCES",
                                "SUBJECT",
                                "question_id"],
                        )

    df.to_pickle('process_raw_question_1.pkl')



def preprocess_deduplication(sample_questions, id_list):
    make_pkl_file(sample_questions, id_list)
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    match_threshold = 0.3
    num_perm = 256
    start_time = datetime.now()
    data_directory = "./raw_data"
    pair_directory = "./pair_data"
    jaccard_directory = "./jaccard_data"

    for d_path in [data_directory, pair_directory, jaccard_directory]:
        if os.path.isdir(d_path):
            os.system(f"mv {d_path} {d_path}_decommissioned_$(date +%F)")
        os.system(f"mkdir -p {d_path}")
        

    warnings.filterwarnings("ignore")
    csv.field_size_limit(sys.maxsize)
    # print("Threshold: ", match_threshold)

    combine_data = pd.read_pickle("./process_raw_question_1.pkl")
    combine_data["SOURCE"] = "TEMP"
    size = combine_data.shape[0]
    start = 0
    batch_size = 10000
    start_list = list(range(0, size, batch_size))
    end_list = start_list[1:]
    end_list.append(size)
    file_count = 0
    # print("data Loaded Count:", size)
    # print("File batch Size: ", batch_size)

    for start, end in zip(start_list, end_list):
        combine_data[start:end].to_pickle(f"{data_directory}/{start}.pkl")
        file_count += 1
    print("Total File Created: ", file_count)

    lsh = MinHashLSH(threshold=match_threshold, num_perm=num_perm)
    # preprocess the data
    files = list(os.listdir(data_directory))
    for file in files:
        # print("-" * 50)
        # print("File Name: ", file)
        data = pd.read_pickle(f"{data_directory}/{file}")
        data["qs_0"] = data.apply(lambda d: text_split(_text=str(d["clean_question"]), subject=str(d["SUBJECT"])), axis=1)
        data["qs_1"] = data.clean_question.apply(text_split_2)
        data["qs_6"] = data.clean_question.apply(text_split_6)
        data["qs_8"] = data.clean_question.apply(remove_digits_formulae)
        data["ns_0"] = data.clean_question.apply(extract_numbers)
        data["nn_1"] = data.clean_question.apply(remove_nouns)
        data["nn_3"] = data.clean_question.apply(noun_detection)
        data["km_0"] = data.clean_question.apply(lambda x: x.lower())
        data["hash"] = data["qs_0"].apply(min_hash_set)
        data["ID"] = data["ID"].astype(str)
        for row in tqdm(data.to_dict(orient="records")):
            lsh.insert(row["ID"], row["hash"])
        data.to_pickle(f"{data_directory}/{file}")
    #     print("process completed")
    #     # break
    # print("Hash updated")

    for file in files:
        # print(file)
        data = pd.read_pickle(f"{data_directory}/{file}")
        data["pairs"] = data["hash"].apply(lambda hs: lsh.query(hs))
        data[["ID", "pairs"]].to_pickle(f"./{pair_directory}/process_{file}")
        
    combine_df = pd.DataFrame()
    files = list(os.listdir(data_directory))
    for file in files:
        data = pd.read_pickle(f"{data_directory}/{file}")
        data = data[['ID', 'url', 'clean_question', 'qs_0', 'qs_1', 'qs_6', 'qs_8', 'ns_0','km_0','nn_1','nn_3','SOURCE','question_id']]
        combine_df = pd.concat([combine_df, data], axis=0)
        
    for file in tqdm(os.listdir(pair_directory)):
        data = pd.read_pickle(f"{pair_directory}/{file}")
        data = data.explode("pairs")
        data = data[data["ID"] != data["pairs"]]
        data.columns = ["ID1", "ID2"]
        data = data.merge(
            combine_df.rename(columns={col: f"{col}1" for col in combine_df.columns}),
            how="left",
            on="ID1"
        ).merge(
            combine_df.rename(columns={col: f"{col}2" for col in combine_df.columns}),
            how="left",
            on="ID2"
        )
        data.dropna(subset=["qs_01", "qs_02"], axis=0, inplace=True)
        if not data.shape[0]:
            print("empty df")
        data["jac_sim_0"] = data.apply(lambda x: jac_sim(x, "qs_01", "qs_02"), axis=1)
        data["jac_sim_1"] = data.apply(lambda x: jac_sim(x, "qs_11", "qs_12"), axis=1)
        data["jac_sim_6"] = data.apply(lambda x: jac_sim(x, "qs_61", "qs_62"), axis=1)
        data["jac_sim_8"] = data.apply(lambda x: jac_sim(x, "qs_81", "qs_82"), axis=1)
        data["jac_sim_9"] = data.apply(lambda x: jac_sim(x, "nn_11", "nn_12"), axis=1)
        data["jac_sim_11"] = data.apply(lambda x: jac_sim(x, "nn_31", "nn_32"), axis=1)
        data["jac_sim_13"] = data.apply(lambda x: jac_sim(x, "ns_01", "ns_02"), axis=1)
        
        threshold_k = 0.6
        condition = (data["jac_sim_0"] >= threshold_k) | (data["jac_sim_1"] >= threshold_k) | \
                        (data["jac_sim_6"] >= threshold_k) | (data["jac_sim_8"] >= threshold_k) | \
                        (data["jac_sim_9"] >= threshold_k)
        data_temp = data[condition]
        # kmp_data = data[~condition]
        
        # kmp_data["kmp_sim_0"] = kmp_data.apply(lambda x: kmp_algorithm(x, "km_01", "km_02"), axis=1)
        # kmp_data = kmp_data[kmp_data["kmp_sim_0"] == True]
        # kmp_data["DUP_TYPE"] = "subquestion_type"
        data_temp["q1_inter"] = data_temp.apply(lambda x: len(s1_intersect(x, "qs_01", "qs_02")), axis=1)
        data_temp["q2_inter"] = data_temp.apply(lambda x: len(s2_intersect(x, "qs_01", "qs_02")), axis=1)
        func = "WRatio"
        data_temp[f'{func}'] = data_temp[['clean_question1', 'clean_question2']].apply(
            lambda x: getattr(fuzz, func)(*x), axis=1) / 100
        func = 'token_set_ratio'
        data_temp[f'{func}'] = data_temp[['clean_question1', 'clean_question2']].apply(
            lambda x: getattr(fuzz, func)(*x), axis=1) / 100
        data_temp['f1'] = data_temp[['jac_sim_0', 'jac_sim_1', 'jac_sim_6', 'token_set_ratio', 'WRatio']].mean(axis=1)
        data_temp['f2'] = data_temp[['jac_sim_8']].mean(axis=1)
        data_temp['f3'] = data_temp[['jac_sim_9']].mean(axis=1)
        data_temp['f_noun'] = data_temp[['jac_sim_11']].mean(axis=1)
        data_temp['f_num'] = data_temp[['jac_sim_13']].mean(axis=1)

        data_temp["PREDICT"] = data_temp.apply(get_preds, axis=1)
        data_temp["DUP_TYPE"] = data_temp["PREDICT"].apply(lambda x: x[1][0])
        data_temp["PREDICT"] = data_temp["PREDICT"].apply(lambda x: x[0])
        # data["math_jac"] = data.apply(lambda x: len(s2_intersect(x, "ns_01", "ns_02")), axis=1)
        # data["math_jac"] = data["math_jac"].fillna(0)
        # data = data[(data["math_jac"] > 0.6) | (data["PREDICT"] == 1)]
        data_temp = data_temp[data_temp["PREDICT"] == 1]
        # data = pd.concat([kmp_data, data_temp], axis=0)
        # mathematics = data[data.SUBJECT1.isin(MATHS_LIST) & data.SUBJECT2.isin(MATHS_LIST)]
        # non_mathematics = data[~(data.SUBJECT1.isin(math_list) & data.SUBJECT2.isin(math_list))]
        # non_mathematics["DQI"] = 1
        # # mathematics["math_jac"] = mathematics.apply(lambda x: jac_sim(x, "que_num1", "que_num2"), axis=1)
        # mathematics["math_jac"] = mathematics.apply(lambda x: len(s2_intersect(x, "ns_01", "ns_02")), axis=1)
        # mathematics["math_jac"] = mathematics["math_jac"].fillna(0)
        # non_mathematics["DQI"] = mathematics["math_jac"].apply(lambda x: 1 if x>0.6 else 2)
        # # mathematics = mathematics[mathematics["math_jac"] > 0.6]
        # data = pd.concat([non_mathematics, mathematics], axis=0)
        data_temp.to_pickle(f"./{jaccard_directory}/{file}")

    print("Pair's Complied, Time taken: ", datetime.now() - start_time)

def find_duplicate_questions():
    file_path = './jaccard_data/process_0.pkl'
    df = pd.read_pickle(file_path)
    data = df[["clean_question1","clean_question2","DUP_TYPE","question_id1","question_id2"]]
    return data


def de_dup(sample_questions:list, id_list: list) -> list:
    # sample_questions = [html2text.html2text(q).replace("\n","") for q in sample_questions]
    preprocess_deduplication(sample_questions, id_list)
    return find_duplicate_questions()


if __name__ == "__main__":
    qns = [
  '1) The Short form of $20,00,000+70,000+300+8=$ a) $20,70,308$ b) $2,70,380$ c) $20,07,308$',
  '3) Predecessor of 99,890 - successor of $23,156=$ a) 76,732 b) 76,789 c) 76,734',
  '6) When $24,01,398+1000$, the remainder will be a) 240 b) 1398 c) 398',
  '12) The Roman numeral for the number 25 is ',
  '14) The measure of a straight angle is ',
  '15) Arun is facing towards the east direction, if he turns two right angles towards his left, then the direction he is facing now is ',
  'Find the values of x, y, z in the following figure.',
  'An exterior angle is 130° and its interior opposite angles are equal. Then find the measure of each interior angle of a triangle.',
  'Maths teacher draws a straight line AB shown on the blackboard as per the following figure. Now he told Raju to draw another line CD as in the figure, then the teacher asked Ajay and Suraj to mark ∠AOD as 2z, and ∠AOC as 4y respectively, Ramya Made an angle ∠COE=60°, Peter marked ∠BOE and ∠BOD as y and x respectively. (i) What is the value of x? (ii) What is the value of y? (iii) What is the value of z? (iv) What should be the value of x+2z?',
  'The ratio of 75 cm to 5 m is:',
  'A study tour is being planned for a class having 35 students. Only 80% are going for the tour. How many students are not going for the tour?',
  'A shirt marked at Rs 840 is sold for Rs 714. What is the discount and discount percent?',
  'Find the cube root by prime factorization method: 13824',
  'A shopkeeper sold two almirahs for Rs. 30,000 each. On one he made a gain of 20% and on the other he loses 20%. Find his gain or loss percent on the whole transaction.',
  'The length of a parallelogram exceeds its breadth by 30 cm. If the perimeter of the parallelogram is 2 m 60 cm, find the length and breadth of the parallelogram.',
  'Simplify (3a-2)(a-1)(3a+5)-9a^3-19a+10',
  'If x+1/x=8, find the value of x^2+1/x^2',
  'What must be added to 10a^2b+8ab^2-8a^3b^3-b^4 to get 5a^2b-6ab^2-7a^3b^3?',
  'The population of a town increased to 54000 in 2011 at a rate of 5% per annum. What was the population in 2009? What will be its population in 2013?',
  'Name each of the following parallelograms: a) The diagonals are equal and the adjacent sides are equal. b) All the angles are equal and the adjacent sides are unequal. c) All the sides are equal and one angle is $60^{\\circ}$.',
  'In the adjacent figure, the bisections of $L A$ and $L B$ meet in a point $P$. If $\\angle C=100^{\\circ}$ and $L D=60^{\\circ}$, find the measure of $\\angle A P B$.',
  'The following table shows the expenditure in percentage incurred in the construction of a house in a city. \\begin{tabular}{|l|l|l|l|l|l|}\\hline Item & Brick & Cement & Steel & Labour & Miscellaneous \\\\ \\hline $\\begin{array}{l}\\text { Expenditure in } \\\\ \\text { percentage }\\end{array}$ & $15 \\%$ & $20 \\%$ & $10 \\%$ & $25 \\%$ & $30 \\%$ \\\\ \\hline\\end{tabular} Represent the above data by a pie chart.',
  'Represent $\\frac{13}{5}$ and $-\\frac{13}{5}$ on the number lines.',
  'Factorise $(1+m)^{2}-(1-m)^{2}$',
  'Find the angles $x, y$ and $z$.',
  '2. Find the odd one out.',
  'In a triangle PQR, if 2∠p=3∠Q=6∠R then find the measure of all the angles.',
  'Find the value of P, Q, and r in the below-given figure.',
  'A scalene triangle has',
  'If we rotate a right-angled triangle of height 5 cm and base 3 cm about its base, we get',
  'In standard form 72 crore is written as',
  'Each of the two equal angles of a triangle is twice the third angle. Find the angles.',
  'Assertion: The integers on the number line form an infinite sequence. Reason: A list of numbers following a definite rule which goes on forever is called an infinite sequence.',
  'What should be added to -3 to get -14/15?',
  'The vertical angle of an isosceles triangle is 15° more than each of its base angle. Find each angle of the triangle.',
  'One angle of a triangle is 60°. The other two angles are in the ratio 5:7. Find the two angles.',
  'Divide the sum of 65/9 and -11/3 by the product of 7/6 and -5/3.',
  '15. If $\\frac{9^{n} \\times 3^{2} \\times 3^{4}-(27)^{n}}{\\left(3^{3}\\right)^{5} \\times 2^{3}}=\\frac{1}{27}$ Find the value of $n$.',
  '16. The length of the diagonals of a rhombus are $12 \\mathrm{~cm}$ and $16 \\mathrm{~cm}$. Calculate the perimeter of the rhombus.',
  '17. A ladder $15 \\mathrm{~m}$ long reaches a window which is $9 \\mathrm{~m}$ above the ground on one side of a street. Keeping its foot at the same point, the ladder is turned to the other side of the street to reach a window $12 \\mathrm{~m}$ high. Find the width of the street.',
  '19. Suresh is having a garden near Delhi. In the garden there are different types of trees and flower plants. One day due to heavy rain and storm one of the trees got broken as shown in the figure. The height of the unbroken part is $15 \\mathrm{~m}$ and the broken part of the tree has fallen at $20 \\mathrm{~m}$ away from the base of the tree. Using the Pythagoras answer the following questions.',
  'i. What is the length of the broken part?',
  'ii. What was the height of the full tree?',
  'iii. In the form of a right-angle triangle, what is the length of the hypotenuse?',
  'iv. What is the perimeter of the formed triangle?',
  '5. The value of ∛(81)^(-2) is',
  '6. The solution of equation x + 2y = 4 will be',
  '7. The value of 5^0 × 2^0 + 3^2 × 3^(-2) is',
  '9. In triangle ABC and triangle PQR, AB = PQ, AC = PR, and BC = QR',
  '10. Find the range of the following data: 11, 20, 12, 14, 10',
  '11. The median of the data: 4, 6, 8, 9, 11 is',
  '12. The arithmetic mean of the first 5 natural numbers is',
  '14. If 3^x = 27, then the value for x is',
  '15. A linear equation is that equation which has degree',
  'Find the values of $x$ and $y$ if : $(5 x-3 y, y-3 x)=(4,-4)$',
  'Solve $:\\left(\\sqrt{\\frac{3}{5}}\\right)^{x+1}=\\frac{125}{27}$',
  'Find the length of $\\mathrm{AB}$',
  '1. Multiply $\\frac{7}{18}$ to the reciprocal of $\\left(\\frac{-5}{13}\\right)$. $\\frac{7}{18}$ ను $\\left(\\frac{-5}{13}\\right)$ యొక్క వ్యుత్కమంతో గుణంచంది.',
  '2. Find the value of the variable in the equation $2 x+3=3 x+2$. కీంది సమీకరణంను సాధించి, చరరాశి విలువను కననగొనుము. $2 x+3=3 x+2$.',
  "4. Find the angle ' $x$ ' in the given figure కకీంది పటాల నుండి కోణం యొక్క కొలత 'x' ను కనుగొనండి.",
  '5. Find a rational number between $\\frac{3}{7}$ and $\\frac{4}{8}$. $\\frac{3}{7}$ మరియు $\\frac{4}{8}$ మధ్యగల అకరణీయసంఖ్యను కనుగొనండి.',
  "6. The present age of Vinay's mother is three times the present age of Vinay. After 5 years, sum of their ages. will be 70 years. Find their present ages. వినయ్ తల్లి ప్రస్తుత వయస్సు వినయ్ పప్రస్తుత వయస్సుకు మూడు రెట్లు. 5 సంవత్సరాల తర్వాత వారి వయస్సుల మొత్తం 70 సంవత్సరాలు వారి ప్రస్తుత వయస్సును కనుగొనండి.",
  '7. $\\mathrm{ABCD}$ is a rectangle. With diagonals $\\mathrm{AC}$ and $\\mathrm{BD}$ intersecting at point $O$. If the length of $A O$ is 6 units find the length of $B D$ $\\mathrm{ABCD}$ ఒక దీర్ఘచతురసము దీని కర్ణములు $\\mathrm{AC}$ మరియు $\\mathrm{BD}$ లు బిందువు $\\mathrm{O}$ వద్ద ఖండించుకుంటాయి. AO కొలత 6 యూనిట్స్ అయినచో BD కొలతను కనుగొనుము.',
  'Using tally marks make a frequency table with intervals as 800-810, 810-820 and so on for the weekly wages (in ₹) of 30 workers in a factory: 830, 835, 890, 810, 835, 836, 869, 845, 898, 890, 820, 860, 832, 833, 855, 845, 804, 808, 812, 840, 885, 835, 835, 836, 878, 840, 868, 890, 806, 840.',
  'Mention the properties of a Rhombus.',
  'Sum of the digits of a two-digit number is 9. When we interchange the digits the new number is 27 greater than the earlier number. Find the number.',
  'Divide Rs. 510 among three workers in such a way that each worker will get in the ratio of 2: 3: 5. Find the share of each person.',
  'Arrange the following rational numbers in the ascending order: 5/4, 2/3, 6/8, 8/5, 9/2.',
  'Using appropriate properties solve the given below: (2/5) × (-3/7) - (1/14) - (3/7) × (3/5)',
  'In the following figure ABCD is a trapezium in which AB // DC. Find the measure of angle C.',
  'Can a quadrilateral ABCD be a parallelogram if angle D + angle B = 180°?',
  'Can a quadrilateral ABCD be a parallelogram if AB = DC = 8 cm, AD = 4 cm, and BC = 4.4 cm?',
  'Can a quadrilateral ABCD be a parallelogram if angle A = 70° and angle C = 65°?',
  'Construct a quadrilateral ABCD given AB=5.1 cm, AD=4 cm, BC=2.5 cm, ∠A=60°, and ∠B=85°.',
  'Construct a parallelogram ABCD with AB=3.5 cm, BC=4.5 cm, and diagonal BD=5.5 cm.',
  "The multiplicative inverse of '-2' is?",
  'Which of the following is a rational number between 1/3 and 5/3?',
  'For what value of x, -12/x is equal to 3?',
  'Which among the following has the most number of diagonals?',
  'What is the measure of any exterior angle of a regular nonegon?',
  'The solution of the equation 3/7 + x = 17/7 is?',
  'If two adjacent sides of a quadrilateral are equal then the quadrilateral is known as?',
  'If you draw a quadrilateral and then draw all its diagonals, then how many triangles can you find?',
  '$42(4+2)=(42 \\times 4)+(42+2)$ is an example of',
  'In parallelogram ABCD, which of the two angles are equal?',
  'A quadrilateral has two of its angles as $30^{\\circ}$ and $40^{\\circ}$. What type of quadrilateral can it be?',
  'Which of the following equations is a linear equation in one variable?',
  'For how many hours did the maximum number of students watch TV?',
  'How many students watched TV for less than 4 hours?',
  'If 20 people liked classical music, how many young people were surveyed?',
  'Which type of music is liked by the maximum number of people?',
  'Twenty years from now, Arjun will become three times as old as he is now. Suitable equation for this is:',
  'Two sides of a kite have lengths 7.8 cm and 11 cm. What is the perimeter of the kite?',
  'In a parallelogram ABCD, determine the sum of angles B, C, and D if angle A = 60°.',
  '1. Solve: $17+6 p=9$.',
  '2. Find the perimeter of the parallelogram $A B C D$ in which $A B=13 \\mathrm{~cm}$ and $\\triangle D=7 \\mathrm{~cm}$.',
  '3. A three-dimensional shape with flat polygonal faces, straight edges and sharp corners or vertices is called a .......',
  '4. Write the faces, vertices and edges for Triangular prism.',
  '5. Expand 0.0523 using exponents.',
  '7. Solve: $\\frac{z}{z+15}=\\frac{4}{9}$.',
  '8. Find the measure of each exterior angle of a regular decagon.',
  '9. In the given figure, find the angle measure $x$.',
  "10. Using Euler's formula find faces, when vertices $=6$ and edges $=12$.",
  '11. Find $\\mathrm{x}:(5 / 4)^{-x} \\div(5 / 4)^{-4}=(5 / 4)^{5}$.',
  '12. Express the 6020000000000000 in standard form.',
  '15. Two opposite angles of a parallelogram are $(3 x-2)^{\\circ}$ and $(50-x)^{\\circ}$. Find the measure of each angle of the parallelogram.',
  '16. Simplify: $\\left\\{(2 / 3)^{2}\\right\\}^{3} \\times(1 / 3)^{-4} \\times 3^{-1} \\times 6^{-1}$.',
  '18. Solve: $5 x-2(2 x-7)=2(3 x-1)+\\frac{7}{2}=\\frac{5}{2}$.',
  '19. In the given figure below, $A B C D$ is a parallelogram, in which $A O=5 y+1$ $O C=6 y-1, B O=3 x-1, O D=2(x+1)$. Find $x$ and $y$.',
  'Value of x^0, when x=0 is',
  'If 2^(x-3)=1, then value of x is',
  'If 10^(2y)=25, then 10^y equals',
  'Which of the following has equal diagonals?',
  'Order of rotational symmetry of an equilateral triangle is',
  'Find the value of $(3^{0}+4^{-1}) \\times 2^{2}$',
  'In trapezium PQRS, PS || QR, SR ⊥ QR, ∠Q = 130°. Find measures of ∠P and ∠S.',
  'Find the volume of a cube, whose surface area is 600 cm².',
  'List two letters of English alphabet which exhibit both line as well as rotational symmetry.',
  "Find the value of 'm' for which $5^{m}: 5^{3}=5^{5}$.",
  'The internal measures of the length, breadth and height of a room are 12 m, 8 m and 4 m respectively. Find the cost of white washing all four walls of the room, if the cost of white washing is ₹ 5/m².',
  'In a building there are 24 cylindrical pillars. The radius of each pillar is 28 cm and height of each is 4 m. Find the total cost of painting the curved surface area of all pillars, at the rate of ₹ 8/m².',
  'Simplify $\\frac{3^{-5} \\times 10^{-5} \\times 125}{5^{7} \\times 6^{-5}}$',
  'The Population of a city increases at the rate of 5% P.a. If the population in 2003 is 52920, what was the population in year 2001?',
  'Solve the linear equation (3t-2)/4-(2t+3)/3=2/3-t.',
  'The measures of two adjacent angles of a parallelogram are in the ratio 3:2. Find the measure of each of the angles of the parallelogram.',
  'In parallelogram HOPE, angle HOP=110°, angle EHP=40°. Find the angle measures of angle HEP, angle OHP, angle OPH.',
  'Present ages of Anu and Raj are in the ratio 4:5. Eight years from now the ratio of their ages will be 5:6. Find their Present ages.',
  'The denominator of a fraction is greater than its numerator by 8. If the numerator is increased by 17 and the denominator is decreased by 1, the number obtained is 3/2. Find the fraction.',
  'A pair of adjacent sides of a rectangle are in the ratio 4:3. If its diagonal is 20 cm, Find the lengths of the sides and hence, the Perimeter of the rectangle.',
  'Divide (125-225x+135x^2-27x^3) by (5-3x) and write down the quotient and remainder.',
  'Solve the following linear equation: 3x - 7 = 2x + 5',
  'Find the value of x in the given proportion: 3/4 = x/12',
  'Calculate the area of a rectangle with a length of 8 cm and a width of 5 cm.',
  'Which of the following is the correct formula for finding the volume of a cylinder?',
  'Simplify the following expression: 4x + 3y - 2x + 5y',
  'Solve the following system of linear equations using the substitution method: x + y = 5 and x - y = 1',
  'Which of the following is the correct formula for finding the circumference of a circle?',
  'Evaluate the following expression when x = 2 and y = 3: 5x^2 - 3y + 4',
  'Find the square root of 144.',
  '<p>Multiply&nbsp;<span class="latexEle" data-latex="\\frac{6}{13}">$\\frac{6}{13}$</span><span>\u2009\u2009by the reciprocal of&nbsp;<span class="latexEle" data-latex="\\frac{-7}{16}">$\\frac{-7}{16}$</span><span>\u2009\u2009.</span></span></p>',
  '<p>Verify that :&nbsp;<span class="latexEle" data-latex="-\\left(x+y\\right)=\\left(-x\\right)+\\left(-y\\right),">$-\\left(x+y\\right)=\\left(-x\\right)+\\left(-y\\right),$</span> when</p> <p>(a)&nbsp;&nbsp;<span class="latexEle" data-latex="x=\\frac{3}{4},y=\\frac{6}{7}">$x=\\frac{3}{4},y=\\frac{6}{7}$</span></p> <p>(b)&nbsp;<span class="latexEle" data-latex="x=\\frac{-3}{4},y=\\frac{-6}{7}">$x=\\frac{-3}{4},y=\\frac{-6}{7}$</span></p>    ',
  'In a proper fraction the ____________ shows the number of parts into which the whole is divided.',
  '<p>Write the rational numbers that are their own reciprocals.</p>',
  '<p>Write the additive inverse of:</p> <p><span class="latexEle" data-latex="\\left(a\\right)\\frac{2}{8}">$\\left(a\\right)\\frac{2}{8}$</span></p> <p><span class="latexEle" data-latex="\\left(b\\right)\\frac{-5}{9}">$\\left(b\\right)\\frac{-5}{9}$</span></p> <p><span class="latexEle" data-latex="\\left(c\\right)\\frac{9}{-16}">$\\left(c\\right)\\frac{9}{-16}$</span></p> <p><span class="latexEle" data-latex="\\left(d\\right)\\frac{-21}{-40}">$\\left(d\\right)\\frac{-21}{-40}$</span></p>  ',
  '<p>Simplify:</p> <p>(a)<span class="latexEle" data-latex="\\frac{\\left(-4\\right)}{9}\\times\\frac{3}{5}\\times\\frac{\\left(-9\\right)}{10}">$\\frac{\\left(-4\\right)}{9}\\times\\frac{3}{5}\\times\\frac{\\left(-9\\right)}{10}$</span></p> <p>(b)<span class="latexEle" data-latex="\\frac{\\left(-11\\right)}{7}\\times\\frac{4}{40}\\times\\frac{21}{33}">$\\frac{\\left(-11\\right)}{7}\\times\\frac{4}{40}\\times\\frac{21}{33}$</span></p>',
  '<p>Verify that :&nbsp;<span class="latexEle" data-latex="-\\left(x+y\\right)=\\left(-x\\right)+\\left(-y\\right),">$-\\left(x+y\\right)=\\left(-x\\right)+\\left(-y\\right),$</span> when</p> <p>(a)&nbsp;&nbsp;<span class="latexEle" data-latex="x=\\frac{3}{4},y=\\frac{6}{7}">$x=\\frac{3}{4},y=\\frac{6}{7}$</span></p> <p>(b)&nbsp;<span class="latexEle" data-latex="x=\\frac{-3}{4},y=\\frac{-6}{7}">$x=\\frac{-3}{4},y=\\frac{-6}{7}$</span></p>    ',
  '<p>Find the value of&nbsp;<span class="latexEle" data-latex="-\\frac{5}{9}\\div\\frac{2}{3}">$-\\frac{5}{9}\\div\\frac{2}{3}$</span></p>',
  '<p>Verify&nbsp;<span class="latexEle" data-latex="x+y=y+x,">$x+y=y+x,$</span>&nbsp;if&nbsp;<span class="latexEle" data-latex="x=\\frac{-3}{16}">$x=\\frac{-3}{16}$</span>&nbsp;and&nbsp;<span class="latexEle" data-latex="y=\\frac{1}{9}.">$y=\\frac{1}{9}.$</span>&nbsp;</p>',
  '<p>Write the rational numbers that are their own reciprocals.</p>',
  '<p>Verify that :&nbsp;<span class="latexEle" data-latex="-\\left(x+y\\right)=\\left(-x\\right)+\\left(-y\\right),">$-\\left(x+y\\right)=\\left(-x\\right)+\\left(-y\\right),$</span> when</p> <p>&nbsp;&nbsp;<span class="latexEle" data-latex="x=\\frac{3}{4},y=\\frac{6}{7}">$x=\\frac{3}{4},y=\\frac{6}{7}$</span></p> <p><br></p>    ',
  '<p>What should be added to <span class="latexEle" data-latex="\\left(\\frac{1}{2}+\\frac{1}{3}+\\frac{1}{5}\\right)">$\\left(\\frac{1}{2}+\\frac{1}{3}+\\frac{1}{5}\\right)$</span>&nbsp; to get <span class="latexEle" data-latex="1">$1$</span>?</p>',
  '<p>Verify the following:</p> <p>(a)&nbsp;<span class="latexEle" data-latex="\\frac{5}{7}+\\frac{-12}{5}=\\frac{-12}{5}+\\frac{5}{7}">$\\frac{5}{7}+\\frac{-12}{5}=\\frac{-12}{5}+\\frac{5}{7}$</span>&nbsp;</p> <p>(b)<span class="latexEle" data-latex="\\left(\\frac{-3}{4}+\\frac{17}{8}\\right)+\\frac{-1}{2}=\\frac{-3}{4}+\\left(\\frac{17}{8}+\\frac{-1}{2}\\right)">$\\left(\\frac{-3}{4}+\\frac{17}{8}\\right)+\\frac{-1}{2}=\\frac{-3}{4}+\\left(\\frac{17}{8}+\\frac{-1}{2}\\right)$</span>&nbsp;</p> <p>(c)&nbsp;&nbsp;<span class="latexEle" data-latex="\\left(\\frac{2}{-9}+\\frac{-3}{5}\\right)+\\frac{1}{3}=\\frac{2}{-9}+\\left(\\frac{-3}{5}+\\frac{1}{3}\\right)">$\\left(\\frac{2}{-9}+\\frac{-3}{5}\\right)+\\frac{1}{3}=\\frac{2}{-9}+\\left(\\frac{-3}{5}+\\frac{1}{3}\\right)$</span></p> <p>(d)&nbsp;<span class="latexEle" data-latex="\\left(\\frac{-7}{11}+\\frac{2}{-5}\\right)+\\frac{-13}{12}=\\frac{-7}{11}+\\left(\\frac{2}{-5}+\\frac{-13}{12}\\right)">$\\left(\\frac{-7}{11}+\\frac{2}{-5}\\right)+\\frac{-13}{12}=\\frac{-7}{11}+\\left(\\frac{2}{-5}+\\frac{-13}{12}\\right)$</span></p>  ',
  '<p>The following pie diagram shows all the expenditures incurred in preparation of a book by a publisher, under various heads. Study the diagram carefully and answer the question</p><p>A: paper <span class="latexEle" data-latex="20\\%">$20\\%$</span>&nbsp;<br><br>B: printing <span class="latexEle" data-latex="25\\%">$25\\%$</span>&nbsp;<br><br>C: binding, convening, designing etc.,&nbsp;<span class="latexEle" data-latex="30">$30$</span><span class="latexEle" data-latex="\\%">$\\%$</span>&nbsp; <br><br>D: miscellaneous&nbsp;<span class="latexEle" data-latex="10\\%">$10\\%$</span>&nbsp; <br><br>E: royalty <span class="latexEle" data-latex="15\\%">$15\\%$</span>&nbsp;<br></p><p><img class="richTextImage" src="https://assessed.co.in:3050/1491203215811-1490704574672-1466497196014-000.png" style="width: 25%;"><br></p>   <div class="image-caption-holder"></div><p><br></p><p>What is the angle of pie diagram showing the expenditure incurred on paying the royalty?<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</p> <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</p>',
  '<p>What should be added to <span class="latexEle" data-latex="\\left(\\frac{1}{2}+\\frac{1}{3}+\\frac{1}{5}\\right)">$\\left(\\frac{1}{2}+\\frac{1}{3}+\\frac{1}{5}\\right)$</span>&nbsp; to get <span class="latexEle" data-latex="1">$1$</span>?</p>',
  '<p>Multiply&nbsp;<span class="latexEle" data-latex="\\frac{6}{13}">$\\frac{6}{13}$</span><span>\u2009\u2009by the reciprocal of&nbsp;<span class="latexEle" data-latex="\\frac{-7}{16}">$\\frac{-7}{16}$</span><span>\u2009\u2009</span></span></p>',
  '<p>If <span class="latexEle" data-latex="a=\\frac{2}{3}\\:,\\:\\:b=\\frac{5}{4}\\:,\\:\\:c=-\\frac{2}{5\\:}">$a=\\frac{2}{3}\\:,\\:\\:b=\\frac{5}{4}\\:,\\:\\:c=-\\frac{2}{5\\:}$</span>&nbsp;, find the values of a-(b-c) and (a-b)-c.</p>',
  '<p>Simplify:</p> <p>(a)&nbsp;<span class="latexEle" data-latex="\\frac{4}{13}+\\left(\\frac{-6}{7}\\right)">$\\frac{4}{13}+\\left(\\frac{-6}{7}\\right)$</span>&nbsp;</p> <p>(b)&nbsp;<span class="latexEle" data-latex="\\frac{-5}{19}+\\frac{-2}{17}">$\\frac{-5}{19}+\\frac{-2}{17}$</span></p>'
]
    qns = [html2text.html2text(q).replace("\n","") for q in qns]
    qns = set(list(qns))
    duplicate_questions = de_dup(qns)
    print(duplicate_questions)
    
    # final_ques = []
    # de_dup()
    # data = pd.read_pickle("./jaccard_data/process_0.pkl")["clean_question1"].tolist()
    # data = list(set(data))
    # final_ques = data.copy()
    # for ques in qns:
    #     if ques not in data:
    #         final_ques.append(ques)
    
    
    
    