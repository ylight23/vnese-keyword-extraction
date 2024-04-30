import yake
import string
import math
import statistics
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Step 1: Use YAKE to extract candidate keywords


def extract_keywords(text):
    custom_kw_extractor = yake.KeywordExtractor(
        lan="vn", n=5, dedupLim=0.9, dedupFunc='seqm', windowsSize=3, top=20, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    return [keyword[0] for keyword in keywords]

# Step 2: Text preprocessing


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


stop_words = []
with open('vietnamese-stopwords.txt', encoding='utf8') as f:
    for line in f:
        stop_words.append(line.strip())

# Loại bỏ từ dừng


def remove_stopwords(o_sen):
    words = [word for word in o_sen.split() if word not in stop_words]
    return " ".join(words)
# Step 3: Scoring candidates


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

    relatedness = 1 + ((WR + WL) * count_w / max_count) + PL + PR
    return relatedness


def calculate_different(candidate, sentences):
    return sum([1 for sentence in sentences if candidate in sentence])

def calculate_scores(candidates, text, sentences):
    scores = {}
    for candidate in candidates:
        words = candidate.split()
        count_candidate = text.count(candidate)
        if count_candidate == 0:
            casing = 0
        else:
            casing = max(text.count(candidate.capitalize()), text.count(
                candidate.upper())) / (1 + math.log(count_candidate))
        positions = [sentences.index(sentence)
                     for sentence in sentences if candidate in sentence]
        if positions:
            position = math.log(math.log(3 + statistics.median(positions)))
        else:
            position = 0
        frequency = count_candidate / (statistics.mean([text.count(word) for word in text.split(
        )]) + statistics.stdev([text.count(word) for word in text.split()]))
        relatedness = calculate_relatedness(candidate, text, text)
        different = calculate_different(candidate, sentences)

        # Kiểm tra different có bằng 0 không trước khi thực hiện phép chia
        if different == 0:
            score = 0
        else:
            score = (casing * position) / (frequency + relatedness / different)

        scores[candidate] = score
    return scores

# Step 4: Post-processing


def post_process_keywords(keywords):
    # Remove duplicates or synonyms
    return list(set(keywords))

# Main function


def main():
    # Read input text
    with open('input_line.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # Split text into sentences
    sentences = text.split('.')

    # Step 1: Extract candidate keywords using YAKE
    preprocessed_text = preprocess_text(text)
    text_without_stopwords = remove_stopwords(preprocessed_text)
    # Step 2: Preprocess the text
    candidates = extract_keywords(text_without_stopwords)
    
    # Step 3: Calculate scores for candidates
    scores = calculate_scores(candidates, preprocessed_text, sentences)
    
    # Step 4: Post-process keywords
    keywords = post_process_keywords(list(scores.keys()))
    
    # Display keywords with their scores
    for keyword in keywords:
        print("Keyword:", keyword)
        print("Score:", scores[keyword])
    
    
if __name__ == "__main__":
    main()
