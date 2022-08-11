import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
from numpy import linalg as la
from rouge import Rouge


def PageRank(A, alpha):
    M = len(A)
    N = len(A[0])
    for i in range(M):
        if (np.array_equal((A[:, i]), np.zeros(M))):
            A[:, i] = np.ones(M) / M
    B = alpha * A + (1 - alpha) * np.ones((M, M)) / M
    U, D = la.eig(B)
    D = D.real
    X = D[:, 0]
    return X


file = open("story__", "r")
text = file.read()
a_2d_matrix = []
sent_text = nltk.sent_tokenize(text)  # this gives us a list of sentences
nltk_tokens = nltk.word_tokenize(text)  # First Word tokenization

ordered_tokens = set()
result = []
all = []
for word in nltk_tokens:
    if word not in ordered_tokens:
        ordered_tokens.add(word)
        result.append(word)

G = nx.DiGraph()
all = sent_text + result
a_2d_matrix = np.zeros((len(all), len(all)))
sentences_nodes = []

for i in range(0, len(sent_text)):
    for j in range(len(sent_text), len(all)):
        if (result[j - len(sent_text)] in sent_text[i]):
            a_2d_matrix[i][j] = 1
            G.add_edge(i, result[j - len(sent_text)])
    sentences_nodes.append(i)

for i in range(len(sent_text), len(all)):
    for j in range(0, len(sent_text)):
        if (result[i - len(sent_text)] in sent_text[j]):
            a_2d_matrix[i][j] = 1
            G.add_edge(result[i - len(sent_text)], j)

# draw graph
pos = nx.bipartite_layout(G, nodes=sentences_nodes)
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
