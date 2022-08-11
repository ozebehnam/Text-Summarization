import os.path

import nltk
from gensim import corpora
from gensim.models import LsiModel
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from rouge import Rouge

nltk.download('stopwords')
nltk.download('punkt')
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


def load_data(path, file_name):
    """
    Input  : path and file_name
    Purpose: loading text file
    Output : list of paragraphs/documents and
             title(initial 100 words considred as title of document)
    """
    documents_list = []
    titles = []
    with open(os.path.join(path, file_name), "r") as fin:
        for line in fin.readlines():
            text = line.strip()
            documents_list.append(text)
    # print("Total Number of Documents:",len(documents_list))
    titles.append(text[0:min(len(text), 5)])
    return documents_list, titles


def preprocess_data(doc_set):
    """Input  : document list
    Purpose: preprocess text (tokenize, removing stopwords, and stemming)
    Output : preprocessed text """
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts


def prepare_corpus(doc_clean):
    """
    Input  : clean document
    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    Output : term dictionary and Document Term Matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    return dictionary, doc_term_matrix


def create_gensim_lsa_model(doc_clean, number_of_topics, words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word=dictionary)  # train model
    # print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel


def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word=dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def plot_graph(doc_clean, start, stop, step):
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix, doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


number_of_topics = 2
words = 20
document_list, titles = load_data("", "story__")
clean_text = preprocess_data(document_list)
dict1, doc_term_matrix = prepare_corpus(clean_text)
model = create_gensim_lsa_model(clean_text, number_of_topics, words)
corpus_lsi = model[doc_term_matrix]

start, stop, step = 2, 12, 1


# plot_graph(clean_text,start,stop,step)


# for doc, as_text in zip(corpus_lsi, document_list):
#     print(doc, as_text)

def takenext(elem):
    """
    sort
    """
    return elem[1]


# sort each vector by score
vecsSort = list(map(lambda i: list(), range(2)))
for i, docv in enumerate(corpus_lsi):
    for sc in docv:
        isent = (i, abs(sc[1]))
        vecsSort[sc[0]].append(isent)
vecsSort = list(map(lambda x: sorted(x, key=takenext, reverse=True), vecsSort))

# print(vecsSort)

sentIndexes = set()


def selectTopSent(summSize, numTopics, sortedVec):
    topSentences = []
    sent_no = []
    sentInd = set()
    sCount = 0
    for i in range(summSize):
        for j in range(numTopics):
            vecs = sortedVec[j]
            si = vecs[i][0]
            if si not in sentInd:
                sent_no.append(si)
                topSentences.append(vecs[i])
                sentInd.add(si)
                sCount += 1
                if sCount == summSize:
                    return sent_no


def selectTopSent(summSize, numTopics, vecsSort):
    topSentences = []
    sent_no = []
    sentIndexes = set()
    sCount = 0
    for i in range(summSize):
        for j in range(numTopics):
            vecs = vecsSort[j]
            si = vecs[i][0]
            if si not in sentIndexes:
                sent_no.append(si)
                sCount += 1
                # print("vecs", vecs[i])
                # print("index", si)
                topSentences.append(vecs[i])
                sentIndexes.add(si)
                if sCount == summSize:
                    sent_no
        return sent_no


topSentences = selectTopSent(8, 2, vecsSort)
topSentences.sort()

summary = []
doc = []
cnt = 0
for sentence in document_list:
    doc.append(sentence)
    if cnt in topSentences:
        summary.append(sentence)
    cnt += 1
summary = " ".join(summary)
doc = " ".join(doc)
print(summary)

file = open("refrence", "r")
reference = file.read()
rouge = Rouge()
scores = rouge.get_scores(summary, reference)
print(scores)
