import yake
import os
import string
import math
import statistics
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Tiền xử lý văn bản


def preprocess_text(text):
    # Chuyển văn bản về chữ thường
    text = text.lower()
    # Loại bỏ các ký tự đặc biệt và dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Loại bỏ từ dừng


def remove_stopwords(o_sen):
    words = [word for word in o_sen.split() if word not in stop_words]
    return " ".join(words)

# Tính toán mức độ liên quan giữa từ khóa và ngữ cảnh xung quanh


def calculate_relatedness(word, left_context, right_context):
    left_words = set(left_context.split())
    right_words = set(right_context.split())

    if len(left_context.split()) == 0:
        WL = 0
    else:
        WL = len(left_words) / len(left_context.split())

    if len(right_context.split()) == 0:
        WR = 0
    else:
        WR = len(right_words) / len(right_context.split())

    max_count = max(len(left_context.split()), len(right_context.split()))
    PL = len(left_context.split()) / max_count
    PR = len(right_context.split()) / max_count

    count_w = (left_context + right_context).split().count(word)
    max_count = max(len(left_context.split()), len(right_context.split()))

    relatedness = 1 + (WR + WL) * count_w / max_count + PL + PR
    return relatedness

# Tính số câu mà từ khóa xuất hiện


def calculate_different(keyword, sentences):
    return sum([1 for sentence in sentences if keyword in sentence])

# Tính điểm số cuối cùng của từ khóa


def calculate_final_score(scores, keyword):
    # Điểm số cuối cùng của từ khóa
    final_score = 1
    for score in scores:
        final_score *= score

    # Tính điểm số của từ khóa ứng viên
    keyword_score = final_score / (1 + sum(scores)) * keyword.count(keyword)
    return keyword_score


# Đọc từ dừng từ tập tin
stop_words = []
with open('vietnamese-stopwords.txt', encoding='utf8') as f:
    for line in f:
        stop_words.append(line.strip())

# Đọc văn bản từ tập tin đầu vào
text = ''
with open('input_line.txt', encoding='utf8') as f:
    text = f.read()

# Tiền xử lý văn bản
preprocessed_text = preprocess_text(text)

# Loại bỏ từ dừng từ văn bản đầu vào
text_without_stopwords = remove_stopwords(preprocessed_text)

# Đặt thông số cho trích xuất từ khóa
language = "vn"
max_ngram_size = 3
deduplication_thresold = 0.6
deduplication_algo = 'seqm'
window_size = 1
num_of_keywords = 20

# Tạo trích xuất từ khóa tùy chỉnh
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                            dedupFunc=deduplication_algo, windowsSize=window_size, top=num_of_keywords, features=None)

# Trích xuất các từ khóa từ văn bản
keywords = custom_kw_extractor.extract_keywords(text_without_stopwords)

# Split text into sentences
sentences = text.split('.')
sentences.extend(text.split('!'))
sentences.extend(text.split('?'))

# Remove empty strings and strip whitespace
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

different_w = {}
for keyword in keywords:
    different_w[keyword[0]] = calculate_different(keyword[0], sentences)

# Tính điểm số cuối cùng của mỗi từ khóa
scores = {}
for keyword in keywords:
    # Tính điểm của mỗi từ
    casing = max(text_without_stopwords.count(keyword[0].capitalize()), text_without_stopwords.count(
        keyword[0].upper())) / (1 + math.log(text_without_stopwords.count(keyword[0])))
    positions = [sentences.index(sentence)
                 for sentence in sentences if keyword[0] in sentence]
    if positions:
        position = math.log(math.log(3 + statistics.median(positions)))
    else:
        position = 0
    frequency = text_without_stopwords.count(keyword[0]) / (statistics.mean([text_without_stopwords.count(
        word) for word in text_without_stopwords.split()]) + statistics.stdev([text_without_stopwords.count(word) for word in text_without_stopwords.split()]))
    relatedness = calculate_relatedness(
        keyword[0], text_without_stopwords, text_without_stopwords)
    different = different_w[keyword[0]]

    final_score = casing + position + frequency + relatedness + different
    scores[keyword[0]] = final_score

# Hiển thị các từ khóa và mức độ liên quan của chúng với ngữ cảnh xung quanh
for keyword in keywords:
    print("Keyword:", keyword[0])
    print("Different:", different_w[keyword[0]])
    print("Final Score:", scores[keyword[0]])
# Generate WordCloud
words = [kw for kw, _ in keywords]
wordcloud = WordCloud().generate(' '.join(words))

# Display the WordCloud
# plt.figure(figsize=(10, 10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()
