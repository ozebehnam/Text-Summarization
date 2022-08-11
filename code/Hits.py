# importing modules
import math

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import pandas as pd
from nltk.corpus import stopwords
from rouge import Rouge


def __is_punctuation(word):
    return word in [".", "?", "!", ",", "\"", ":", ";", "'", "-"]


def __remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words and not __is_punctuation(i)])
    return sen_new


def __solve(s0, s1):
    s0 = s0.lower()
    s1 = s1.lower()
    s0List = s0.split(" ")
    s1List = s1.split(" ")
    return len(list(set(s0List) & set(s1List)))


file = open("story__", "r")
text = file.read()
sent_text = nltk.sent_tokenize(text)
clean_sentences = pd.Series(sent_text).str.replace("[^a-zA-Z]", " ", regex=True)
clean_sentences = [s.lower() for s in clean_sentences]
stop_words = stopwords.words('english')
clean_sentences = [__remove_stopwords(r.split()) for r in clean_sentences]
G = nx.DiGraph()
for i, x in enumerate(clean_sentences):
    for j, y in enumerate(clean_sentences):
        if (i != j):
            sim = __solve(x, y) / (math.log(len(x)) + math.log(len(y)))
            if sim >= 0.3:
                G.add_edge(i, j)

hubs, authorities = nx.hits(G, max_iter=50, normalized=True)

# draw graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

highest_value = sorted(authorities.items(), key=lambda x: x[1], reverse=True)
important_sent = ""
for i in range(5):
    print(sent_text[highest_value[i][0]])
    important_sent += sent_text[highest_value[i][0]]

hypothesis = important_sent
file = open("refrence", "r")
reference = file.read()
rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)
