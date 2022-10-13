import pickle

from collections import Counter
import numpy as np
import pandas as pd

class PoSTagger:

    def __init__( self, corpus ):
        self.corpus = corpus
        self.vocab  = set()
        self.tagset = set()
        self.p_wt_df = []       # dataframe probability of word i with tag j

    def fit( self ):
        # creating the vocabulary and set of tags
        for sentence in self.corpus:
            for word, tag in sentence:
                self.vocab.add(word)
                self.tagset.add(tag)

    def train( self ):
        # lines = words | columns = tags
        wt_mtx = np.zeros( (len(self.vocab), len(self.tagset)), dtype='float32' )

        # ordered list of the pairs (word, tag)
        pairs_wt = [pair for sentence in self.corpus for pair in sentence]

        # count for each pair (word, tag)
        counter_word_tag = Counter(pairs_wt)
        # count of tags
        _, tags = zip(*pairs_wt)
        counter_tag = Counter(tags)

        tagset_len = len(self.tagset)
        # filling the word-tag matrix
        for i, word in enumerate(list(self.vocab)):
            for j, tag in enumerate(list(self.tagset)):
                count_word_tag = counter_word_tag[(word, tag)]
                count_tag = counter_tag[tag]

                # formula with smoothing
                epsilon = 0.001
                wt_mtx[i][j] = (count_word_tag + epsilon) / (count_tag + tagset_len * epsilon)

        self.p_wt_df = pd.DataFrame(wt_mtx,
                               index = list(self.vocab),
                               columns = list(self.tagset))

    def serialize_matrix( self ):
        # Serialization
        with open("p_wt_df.pickle", "wb") as outfile:
            pickle.dump(self.p_wt_df, outfile)

    def load_matrix( self ):
        # Deserialization
        with open("p_wt_df.pickle", "rb") as infile:
            self.p_wt_df = pickle.load(infile)

    # receives a list of untagged sentences
    def predict( self, sentences ):
        tagged = []

        for sentence in sentences:
            selected_tags = []
            for word in sentence:
                if word in self.vocab:
                    tags = self.p_wt_df.loc[word]      # selecting the row containing the probabilities for each tag
                    tag  = tags.idxmax()               # selecting the index of the max probability
                    selected_tags.append(tag)
                else:
                    selected_tags.append('UNK')
            tagged.append( list(zip(sentence, selected_tags)) )

        return tagged
