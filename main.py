from pos_tagger import PoSTagger

def check_invalid_symb( symb ):
    # regex??
    if (symb != '.') and (symb != ',') and (symb != '#') and (symb != '$') and (symb != '``') and (symb != ':') and (symb != '') and (symb != "''"):
        return 1
    return 0

with open('doc/training_0_18') as f:
    lines = f.readlines()

    word_tags = []   # corpus
    # turning lines into tuples of type (word, tag)
    for line in lines:
        line = line.lower().strip().split(' ')     # cleaning line: lowercased, \n removed
                                                   # and turned into a list
        for elem in line:
            pair = tuple( elem.split('_') )
            # remove invalid tags
            if check_invalid_symb( pair[1] ):
                word_tags.append( pair )

    
    tagger = PoSTagger(word_tags)
    print(tagger.ngrams(2))
