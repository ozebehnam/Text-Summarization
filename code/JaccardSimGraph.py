import math
import networkx as nx
import nltk
import numpy as np
from nltk.corpus import stopwords
from numpy import linalg as la
from rouge import Rouge
import matplotlib.pyplot as plt


def __is_punctuation(word):
    return word in [".", "?", "!", ",", "\"", ":", ";", "'", "-"]


def __is_stopwords(word):
    return word in stopwords.words('english')


def __preprocess_document(document):
    words = nltk.word_tokenize(document)

    # Filter out words
    filtered_words = []
    for index, word in enumerate(words):
        word = word.lower()
        if not __is_punctuation(word) and not __is_stopwords(word):
            filtered_words.append(word)
    sentence = " "
    sentence = sentence.join(filtered_words)
    return sentence

def split_sent(sent):
    return sent.split()

def jaccard_similarity(s0, s1):
    intersection = set(split_sent(s0)).intersection(set(split_sent(s1)))
    union = set(split_sent(s0)).union(set(split_sent(s1)))
    return len(intersection) / len(union)


def PageRank(A, alpha):
    M = len(A)
    for i in range(M):
        if (np.array_equal((A[:, i]), np.zeros(M))):
            A[:, i] = np.ones(M) / M
    B = alpha * A + (1 - alpha) * np.ones((M, M)) / M
    U, D = la.eig(B)
    X = D[:, 0]
    return X


file = open("story__", "r")
text = file.read()
a_2d_matrix = []
sent_text = nltk.sent_tokenize(text)
sent_text_ = [__preprocess_document(s) for s in sent_text]


G = nx.Graph()
for index1, sentence1 in enumerate(sent_text_):
    col = []
    for index2, sentence2 in enumerate(sent_text_):
        if (index1 == index2):
            col.append(0)
        elif (index1 != index2):
            sim = jaccard_similarity(sentence1, sentence2)
            col.append(sim)
            if (sim != 0):
                G.add_edge(index1, index2)
                G[index1][index2]['weight'] = round(sim, 2)
    a_2d_matrix.append(col)

# draw graph
pos = nx.get_edge_attributes(G, 'pos')
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

ranked = PageRank(np.array(a_2d_matrix), 0.85)

important_sent = ""
# sort ranked sentences based on rank
ranked_sentences = sorted(((ranked[i], s, i) for i, s in enumerate(sent_text)), reverse=True)
# sort 5 up-ranked  bassed on index in story
sort_ranked_sentences = sorted(((s[2], s[1]) for i, s in enumerate(ranked_sentences[:5])), reverse=False)

for item in sort_ranked_sentences:
    important_sent += item[1]
    print(item[1])

hypothesis = important_sent
file = open("refrence", "r")
reference = file.read()
rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)
