import re
import sys
from collections import defaultdict
import gensim.downloader
import numpy as np


# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t)  # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t)  # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens


# Similarity metric cosine
def Cosine(val_0, val_1):
    # cosine similarity: a dot b / ||a||*||b|| where ||a|| = sqrt(sum(a^2)), ||b|| = sqrt(sum(b^2))
    return (np.dot(val_0, val_1) / (np.linalg.norm(val_0) * np.linalg.norm(val_1))) * 10  # numpy makes math easy


class Baseline():
    def __init__(self):
        self.pred_sim = 5  # Always predicts the sim

    def calc_sim(self, word_0, word_1):
        return self.pred_sim


# using the count-vector from a term-document matrix
class Term_document():
    def __init__(self, corpus):
        self.learn_embeddings(corpus)

    def learn_embeddings(self, corpus):
        corpus = open(corpus).readlines()
        freq_dict = defaultdict(int)
        n = len(corpus)  # amount of zeroes in each vector -- amount of docs

        doc_index = 0
        for doc in corpus:
            tokenized_doc = tokenize(doc)
            for word in tokenized_doc:
                if word not in freq_dict.keys():
                    freq_dict[word] = [0] * n
                freq_dict[word][doc_index] += 1
            doc_index += 1

        self.embeddings = freq_dict, n

    def calc_sim(self, word_0, word_1):
        if word_0 in self.embeddings[0].keys() and word_1 in self.embeddings[0].keys():
            similarity = Cosine(self.embeddings[0][word_0], self.embeddings[0][word_1])  # similarity score
            return str(similarity)
        else:
            return str(5)


class Window():
    def __init__(self, corpus):
        self.train(corpus)

    def train(self, corpus):
        t = 2  # hyperparameter, used to change the 'window' size
        corpus = open(corpus).readlines()
        # initialize dicts
        word_word_matrix = {}
        word_to_index = defaultdict(int)
        index_to_word = defaultdict(str)
        word_types = set()

        i = 0  # counter
        # initialize word types and word_to_index, index_to_word dicts
        for doc in corpus:
            words = tokenize(doc)
            for word in words:
                word_types.add(word)
        for word in word_types:
            word_to_index[word] = i
            index_to_word[i] = word
            i += 1

        # setup context
        for doc in corpus:
            words = tokenize(doc)
            for i in range(len(words)):
                target_word = words[i]  # set target word
                context = []  # initialize context list
                # iterate through the words within the context window
                for j in range(max(0, i - t), min(len(words), i + t + 1)):  # window range
                    # skip the target word
                    if i != j:
                        context.append(words[j])  # add the context word to the list
                co_occurrences = word_word_matrix.get(target_word, {})  # dictionary of co-occurrences for the target word
                # iterate through the context words and increment their counts in the co-occurrence dictionary
                for word in context:
                    co_occurrences[word] = co_occurrences.get(word, 0) + 1
                word_word_matrix[target_word] = co_occurrences  # store words and the co occurence words

        vocab_size = len(word_types)
        co_occurrence_matrix = np.zeros((vocab_size, vocab_size))  # initialize V x V matrix
        UNK_vector = np.zeros(vocab_size)  # special vector to use for unknown words

        for word, similar_words_dict in word_word_matrix.items():
            if word not in word_to_index:
                # if the word is unknown, skip it
                continue
            i = word_to_index[word]  # get the index of the current word

            # loop over each similar word and its frequency
            for similar_word, frequency in similar_words_dict.items():
                if similar_word not in word_to_index:
                    j = UNK_vector  # If the similar word is unknown, use the UNK vector
                else:
                    j = word_to_index[similar_word]  # get the index of the similar word

                co_occurrence_frequency = frequency  # compute co-occurrence frequency,
                co_occurrence_matrix[i, j] = co_occurrence_frequency  # set the co-occurrence frequency in the matrix

        self.trained = word_word_matrix, word_to_index, co_occurrence_matrix

    def calc_sim(self, target_1, target_2):
        if target_1 in self.trained[0].keys() and target_2 in self.trained[0].keys():
            similarity = Cosine(self.trained[2][self.trained[1][target_1]],
                                self.trained[2][self.trained[1][target_2]])  # similarity score
            return str(similarity)
        else:
            return str(5)


class Word2Vec():
    def __init__(self):
        # pretrained model from Gensim
        google_news = gensim.downloader.load('word2vec-google-news-300')  # dataset for model to be trained on
        self.model = google_news

    def calc_sim(self, word_0, word_1):
        embedding_1, embedding_2 = self.model.get_vector(word_0), self.model.get_vector(word_1)  # get vector embeddings for target words
        similarity = Cosine(embedding_1, embedding_2)
        return similarity


if __name__ == '__main__':

    sys.stdout.reconfigure(encoding='utf-8')
    method = sys.argv[1]

    train_corpus_fname = sys.argv[2]
    test_texts_fname = sys.argv[3]

    test_tuples = [x.strip().split(',') for x in open(test_texts_fname,
                                                      encoding='utf8')]

    if method == 'baseline':
        model = Baseline()

    elif method == 'td':
        model = Term_document(train_corpus_fname)

    elif method == 'window':
        model = Window(train_corpus_fname)

    elif method == 'w2v':
        model = Word2Vec()

    # Run the classify method for each instance
    results = [model.calc_sim(x[0], x[1]) for x in test_tuples]

    # Create output file at given output file name
    # Store predictions in output file
    outFile = sys.argv[4]
    out = open(outFile, 'w', encoding='utf-8')
    for r in results:
        out.write(str(r) + '\n')
    out.close()
