from pathlib import Path
import time

from pos_tagger import PoSTagger

def tokenize( lines_list ):
    sentence_list = []   # a list of sentences (also lists)
    # turning lines into tuples of type (word, tag)
    for line in lines_list:
        # cleaning line: lowercased, \n removed
        # and turned into a list
        line = line.lower().strip().split(' ')

        sentence = []
        for elem in line:
            pair = tuple( elem.split('_') )
            sentence.append( pair )

        sentence_list.append( sentence )

    return sentence_list

def extract_pairs( tagged_sentences ):
    pairs = []
    words = []

    for sent in tagged_sentences:
        for pair in sent:
            pairs.append(pair)
            words.append(pair[0])

    return (pairs, words)

print("------------------------------------------")
print("+                PoS Tagger              +")
print("------------------------------------------")

with open('doc/training_0_18') as train:
    train_lines = train.readlines()
    tagged_train = tokenize(train_lines)

tagger = PoSTagger(tagged_train)
tagger.fit()

print("# TRAINING MODEL")

print(">>> SEARCHING FOR SERIALIZED DATA.")
path = Path('tags_matrix.pickle')
if path.is_file():
    print(">>> FILE FOUND! LOADING...")
    tagger.load_matrix()
    print(">>> LOADED.")
else:
    print(">>> NO FILE FOUND. TRAINING...")
    tagger.train()
    tagger.serialize_matrix()
    print(">>> TRAINED AND SERIALIZED.")

print("# DEVELOPMENT SECTION")

with open('doc/development_19_21') as dev:
    dev_lines = dev.readlines()
    tagged_dev = tokenize(dev_lines)

dev_pairs, dev_words = extract_pairs(tagged_dev)

print(">>> PREDICTING TAGS. THIS MIGHT TAKE A WHILE...")
start = time.time()
pred_dev = tagger.predict(dev_words[:50])
end = time.time()
diff = end-start

print("Time taken in seconds: ", diff)
# accuracy
check = [i for i, j in zip(pred_dev, dev_pairs[:50]) if i == j]
accuracy = len(check)/len(dev_pairs[:50])
print("Accuracy: ", accuracy*100)

print("# FINAL TEST")

with open('doc/testing_22_24') as test:
    test_lines = test.readlines()
    tagged_test = tokenize(test_lines)

test_pairs, test_words = extract_pairs(tagged_test)

pred_test = tagger.predict(test_words[:30])
print(pred_test[:10])
