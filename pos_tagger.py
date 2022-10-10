import pickle

import numpy as np
import pandas as pd

class PoSTagger:

    def __init__( self, corpus ):
        self.corpus = corpus
        self.tagged_words = []
        self.tags = []
        self.tags_matrix = []

    def fit( self ):
        for sentence in self.corpus:
            for pair in sentence:
                self.tagged_words.append(pair)
                self.tags.append(pair[1])

    def emissionP( self, word, tag ):
        tags_list  = [pair for pair in self.tagged_words if pair[1] == tag]
        count_tags = len(tags_list)

        word_tags_list = [pair[0] for pair in tags_list if pair[0] == word]
        count_words    = len(word_tags_list)

        return count_words / count_tags

    def transitionP( self, tag1, tag2 ):
        count_t1   = len([tag for tag in self.tags if tag == tag1])
        count_t2t1 = 0

        for idx in range(len(self.tags)-1):
            if self.tags[idx] == tag1 and self.tags[idx+1] == tag2:
                count_t2t1 += 1

        return count_t2t1 / count_t1

    def train( self ):
        tagset = set(self.tags)
        transition_mtx = np.zeros((len(tagset), len(tagset)), dtype='float32')

        # filling transition matrix
        for i, tag1 in enumerate(list(tagset)):
            for j, tag2 in enumerate(tagset):
                transition_mtx[i][j] = self.transitionP( tag1, tag2 )

        self.tags_matrix = pd.DataFrame(
            transition_mtx,
            columns = list(tagset),
            index=list(tagset)
        )

    def serialize_matrix( self ):
        # Serialization
        with open("tags_matrix.pickle", "wb") as outfile:
            pickle.dump(self.tags_matrix, outfile)

    def load_matrix( self ):
        # Deserialization
        with open("tags_matrix.pickle", "rb") as infile:
            self.tags_matrix = pickle.load(infile)

    def predict( self, words ):
        selected_tags = []
        tags = list(set(self.tags))

        for idx, word in enumerate(words):
            prob = [] # probability of each tag for the word
            for tag in tags:
                if idx == 0:
                    transition_p = self.tags_matrix.loc['.', tag]
                else:
                    transition_p = self.tags_matrix.loc[selected_tags[-1], tag]

                emission_p = self.emissionP(word, tag)
                tag_probability = transition_p * emission_p
                prob.append(tag_probability)

            prob_max = max(prob)
            chosen_tag = tags[prob.index(prob_max)]
            selected_tags.append(chosen_tag)

        return list(zip(words, selected_tags))

