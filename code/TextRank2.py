import math
import os
import collections

import networkx as nx
import numpy
import pandas
import pandas as pd
import nltk
import nltk.tokenize
from nltk.corpus import stopwords
from rouge import Rouge
import matplotlib.pyplot as plt

def __extract_nodes(matrix):
    nodes = set()
    for col_key in matrix:
        nodes.add(col_key)
    return nodes


def __ensure_rows_positive(matrix):
    matrix = matrix.T
    for col_key in matrix:
        if matrix[col_key].sum() == 0.0:
            matrix[col_key] = pandas.Series(numpy.ones(len(matrix[col_key])), index=matrix.index)
    return matrix.T


def __normalize_rows(matrix):
    return matrix.div(matrix.sum(axis=1), axis=0)


def __euclidean_norm(series):
    return math.sqrt(series.dot(series))


# PageRank specific functionality:

def __start_state(nodes):
    if len(nodes) == 0: raise ValueError("There must be at least one node.")
    start_prob = 1.0 / float(len(nodes))
    return pandas.Series({node: start_prob for node in nodes})


def __integrate_random_surfer(nodes, transition_probabilities, rsp):
    alpha = 1.0 / float(len(nodes)) * rsp
    return transition_probabilities.copy().multiply(1.0 - rsp) + alpha

def __draw_graph(matrix):

    G = nx.Graph()
    G = nx.from_pandas_adjacency(matrix)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def power_iteration(transition_weights, rsp=0.15, epsilon=0.00001, max_iterations=1000):
    # Clerical work:
    transition_weights = pandas.DataFrame(transition_weights)
    nodes = __extract_nodes(transition_weights)
    transition_weights = transition_weights.fillna(0.0)
    transition_weights = __ensure_rows_positive(transition_weights)

    # Setup:
    state = __start_state(nodes)
    transition_probabilities = __normalize_rows(transition_weights)
    transition_probabilities = __integrate_random_surfer(nodes, transition_probabilities, rsp)

    __draw_graph(transition_weights)
    # Power iteration:
    for iteration in range(max_iterations):
        old_state = state.copy()
        state = state.dot(transition_probabilities)
        delta = state - old_state
        if __euclidean_norm(delta) < epsilon:
            break

    return state

## TextRank

def __preprocess_document(document, relevant_pos_tags):


    words = __tokenize_words(document)
    pos_tags = __tag_parts_of_speech(words)

    # Filter out words with irrelevant POS tags
    filtered_words = []
    for index, word in enumerate(words):
        word = word.lower()
        tag = pos_tags[index]
        if not __is_punctuation(word) and tag in relevant_pos_tags:
            filtered_words.append(word)

    return filtered_words


def textrank(document, window_size=2, rsp=0.15, relevant_pos_tags=["NN", "ADJ"]):

    # Tokenize document:
    words = __preprocess_document(document, relevant_pos_tags)

    # Build a weighted graph where nodes are words and
    # edge weights are the number of times words cooccur
    # within a window of predetermined size. In doing so
    # we double count each coocurrence, but that will not
    # alter relative weights which ultimately determine
    # TextRank scores.
    edge_weights = collections.defaultdict(lambda: collections.Counter())
    for index, word in enumerate(words):
        for other_index in range(index - window_size, index + window_size + 1):
            if other_index >= 0 and other_index < len(words) and other_index != index:
                other_word = words[other_index]
                edge_weights[word][other_word] += 1.0

    # Apply PageRank to the weighted graph:
    word_probabilities = power_iteration(edge_weights, rsp=rsp)
    word_probabilities = word_probabilities.sort_values(ascending=False)

    return word_probabilities


## NLP utilities

def __ascii_only(string):
    return "".join([char if ord(char) < 128 else "" for char in string])


def __is_punctuation(word):
    return word in [".", "?", "!", ",", "\"", ":", ";", "'", "-"]


def __tag_parts_of_speech(words):
    return [pair[1] for pair in nltk.pos_tag(words)]


def __tokenize_words(sentence):
    return nltk.tokenize.word_tokenize(sentence)


def apply_text_tank(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    document = open(file_path).read()

    keyword_scores = textrank(document)

    return keyword_scores


def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def apply_sentences_score(keyword_scores, title):
    sentence_scores = []
    file = open(title, "r")
    text = file.read()
    sentences = nltk.sent_tokenize(text)
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    tokenizer = nltk.RegexpTokenizer(r'\w+')

    for i in range(len(clean_sentences)):
        score = 0
        tokens = tokenizer.tokenize(clean_sentences[i])
        for token in tokens:
            if keyword_scores.get(token):
                score += keyword_scores.get(token)
        sentence_scores.append(score)
    ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    ranked_sentences_text = ""
    for i in range(5):
        print(ranked_sentences[i][1])
        ranked_sentences_text +=ranked_sentences[i][1]
    return ranked_sentences_text


keyword_scores = apply_text_tank("story__")
sentence_scores = apply_sentences_score(keyword_scores, "story__")

#
file = open("refrence", "r")
reference = file.read()
#
rouge = Rouge()
scores = rouge.get_scores(sentence_scores, reference)
print(scores)


