class PoSTagger:

    def __init__(self, tagged_corpus):
        self.tagged_corpus = tagged_corpus
        self.tags = [tag for word, tag in tagged_corpus]
        #self.tags  = {tag for word, tag in tagged_corpus}
        #self.vocab = {word for word, tag in tagged_corpus}
    
    def ngrams(self, n):
        n_grams = []
        for i in range(len(self.tags)):
            n_grams.append( tuple(self.tags[i: i + n]) )
        return n_grams
