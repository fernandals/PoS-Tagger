from pathlib import Path

import time

from pos_tagger import PoSTagger

def annotate_data( lines_list ):
    sentence_list = []   # a list of sentences (also lists)
    # turning lines into tuples of type (word, tag)
    for line in lines_list:
        # cleaning line: lowercased, \n removed
        # and turned into a list
        line = line.strip().split(' ')

        sentence = [('bos', 'BOS')]
        for elem in line:
            pair_l = elem.split('_')
            pair   = ( pair_l[0].lower(), pair_l[1] )
            sentence.append( pair )
        sentence.append( ('eos', 'EOS') )

        sentence_list.append( sentence )

    return sentence_list

def remove_tags( sentence_list ):
    untagged_list = []
    # removind tags for testing
    for sentence in sentence_list:
        new_sentence = []
        for word, _ in sentence:
            new_sentence.append(word)
        untagged_list.append(new_sentence)

    return untagged_list

print("------------------------------------------")
print("+                PoS Tagger              +")
print("------------------------------------------")

with open('doc/training_0_18') as train:
    train_lines = train.readlines()

tagged_train = annotate_data(train_lines)
tagger = PoSTagger(tagged_train)
tagger.fit()

print(">>> SEARCHING FOR SERIALIZED DATA.")
path = Path('p_wt_df.pickle')
if path.is_file():
    print(">>> FILE FOUND! LOADING...")
    tagger.load_matrix()
    print(">>> LOADED.")
else:
    print(">>> NO FILE FOUND.")
    print("# TRAINING MODEL")

    start = time.time()
    tagger.train()
    end = time.time()
    diff = end-start
    print(" ** Time taken to train (s): ", diff)

    tagger.serialize_matrix()
    print(">>> TRAINED AND SERIALIZED.")

print("# DEVELOPMENT SECTION")

with open('doc/development_19_21') as dev:
    dev_lines = dev.readlines()

tagged_dev   = annotate_data(dev_lines)
untagged_dev = remove_tags(tagged_dev)

print(">>> PREDICTING TAGS. THIS MIGHT TAKE A WHILE...")
start = time.time()
pred_dev = tagger.predict(untagged_dev)
end = time.time()
diff = end-start

print(" ** Time taken in seconds: ", diff)

# -- accuracy
pred_pairs = []
tag_pairs  = []

for pred_sent, tag_sent in zip( pred_dev, tagged_dev ):
    for pred_pair, tag_pair in zip( pred_sent, tag_sent ):
        pred_pairs.append(pred_pair)
        tag_pairs.append(tag_pair)

check = 0
for i, j in zip(pred_pairs, tag_pairs):
    if i == j:
        check += 1

accuracy = check/len(pred_pairs)
print(" --- Accuracy (%): ", accuracy*100)

print("# FINAL TEST")

with open('doc/testing_22_24') as test:
    test_lines = test.readlines()

tagged_test = annotate_data(test_lines)
untagged_test = remove_tags(tagged_test)

pred_test = tagger.predict(untagged_test)
print(pred_test[:3])
