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

    ##########################################################################
    # PART 1
    ##########################################################################
    # Tuning Variables
    df_max = 1.0
    df_min = 1
    features_max = 2000
    a = 1
    v_a = a * features_max

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        # Tuning for max_df and min_df
        #max_df = df_max
        #min_df = df_min
        # Add limit on max_features
        max_features = features_max
    )


    # Create BOW vectors for each set of the data
    vector = vectorizer.fit_transform(imdb_data['review'])
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


    ##########################################################################
    # PART 2
    ##########################################################################
    #get p(y=0) and p(y=1) from training labels
    train_rows = imdb_labels.iloc[0:30000]
    train_rows = pd.Series(train_rows['sentiment'])
    train_rows_pos = train_rows.str.count("positive")
    train_rows_neg = train_rows.str.count("negative")
    num_pos = train_rows_pos.sum(axis = 0, skipna = True)
    num_neg = train_rows_neg.sum(axis = 0, skipna = True)
    p_neg = num_neg/30000
    p_pos = num_pos/30000
    #print(p_neg)
    #print(p_pos)


    #get total # of words and # of word i
    train_rows_pos = np.array([train_rows_pos])
    pos_word_v = np.dot(train_rows_pos, train_v)
    pos_total_words = pos_word_v.sum()
    #print(pos_total_words)
    pos_total_words += v_a


    train_rows_neg = np.array([train_rows_neg])
    neg_word_v = np.dot(train_rows_neg, train_v)
    neg_total_words = neg_word_v.sum()
    #print(neg_total_words)
    neg_total_words += v_a

    #print(neg_total_words)
    #print(pos_total_words)

    pos_word_v = (pos_word_v + a)
    neg_word_v = (neg_word_v + a)

    #print(neg_word_v)
    #print(pos_word_v)
    #Calculate Naive Bayes probability for seeing each word

    pos_P_word_v = np.divide(pos_word_v, pos_total_words)
    neg_P_word_v = np.divide(neg_word_v, neg_total_words)

    print(pos_P_word_v)
    print(neg_P_word_v)


    ##########################################################################
    # PART 3
    ##########################################################################

    # Loop through each validation review
        # Calculate P for pos and neg based on pos_P_word_v and neg_P_word_v
        # Keep higher probability
        # Record Guess for each validation review
    # Compare validation guesses with validation labels
    # Output accuracy

    # Loop through each test review
        # Calculate P for pos and neg based on pos_P_word_v and neg_P_word_v
        # Keep higher probability
        # Record Guess for each test review
    # Print Guesses to file "test-prediction1.csv"

