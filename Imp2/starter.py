import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re

# Importing the dataset
imdb_data = pd.read_csv('IMDB.csv', delimiter=',')
imdb_labels = pd.read_csv('IMDB_labels.csv', delimiter=',')

def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

if __name__ == '__main__':

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        # Tuning for max_df and min_df
        #max_df = 1.0
        #min_df = 1
        # Add limit on max_features
        max_features = 2000
    )

    # fit the vectorizer on the text
    vectorizer.fit(imdb_data['review'])

    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

    # Write vocab to file for testing
    #file = open("vocab.txt","w")
    #for ele in vocabulary:
    #    file.write(ele+'\n')
    #file.close
    #print(vocabulary)

    # Create BOW vectors for each set of the data
    vector = vectorizer.transform(imdb_data.get("review"))
    vectors = vector.toarray()
    # Split the vector for each set
    temp = np.split(vectors, [30000, 40000])
    train_v = temp[0]
    valid_v = temp[1]
    test_v = temp[2]

    # Create arrays for training and validation labels
    temp = np.split(imdb_labels.to_numpy(), [30000])
    train_labels = temp[0]
    valid_labels = temp[1]

    # Learning P(y = 0) and P(y = 1) from training labels
    # Learning P(wi|y = [0,1])

    # Initiallize word count total and vectors for negative and positive
    neg_total_words = 0
    neg_word_v = train_v[0]
    pos_total_words = 0
    pos_word_v = train_v[0]

    # Find length of word vector
    leng = len(neg_word_v)
    a = 1 # Alpha set to 1
    v_a = a * leng # |V| * alpha
    # Set each vector element to add alpha to start (Laplace Smoothing)
    for i in range(leng):
        neg_word_v[i] = a
        pos_word_v[i] = a

    # Initiallive the number of negative and positive reviews to 0
    num_neg = 0
    num_pos = 0

    # Loop through all 30000 training elements
    leng = len(train_labels)
    for i in range(leng):
        lengt = len(neg_word_v)
        if train_labels[i] == "negative":
            num_neg += 1
            # Loop through each word in review vector
            for j in range(lengt):
                # Add word count to total neg words and individual word vector
                neg_total_words += train_v[i][j]
                neg_word_v[j] += train_v[i][j]
        else:
            num_pos += 1
            # Loop through each word in review vector
            for j in range(lengt):
                # Add word count to total pos words and individual word vector
                pos_total_words += train_v[i][j]
                pos_word_v[j] += train_v[i][j]

    # Add |V|a to total word counts (Laplace Smoothing)
    neg_total_words += v_a
    pos_total_words += v_a

    # Calculate probability that the review is negative or positive
    p_neg = num_neg/30000
    p_pos = num_pos/30000

    # Calculate Naive Bayes probability for seeing each word
    #  given the review type
    neg_P_word_v = np.divide(neg_word_v, neg_total_words)
    pos_P_word_v = np.divide(pos_word_v, pos_total_words)

    # Testing print statements
    print("P(y = 0):")
    print(p_neg)
    print("P(y = 1):")
    print(p_pos)
    print(neg_total_words)
    print(neg_word_v)
    print(neg_P_word_v)
    print(pos_total_words)
    print(pos_word_v)
    print(pos_P_word_v)

