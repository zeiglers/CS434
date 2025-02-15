import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import csv

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
    # PART 5
    ##########################################################################
    # Tuning Variables
    df_max = 1.0
    df_min = 1
    features_max = 2000
    a = 1
    v_a = a * features_max
    df_max_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df_min_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    feat_max_range = [1000, 1400, 1800, 2200, 2600]
    optAcc  = 0.0
    optMax  = 0.0
    optMin  = 0
    optFeat = 0
        
    for df_max in df_max_range:
        for df_min in df_min_range:

            #create prune room
            if(df_max - df_min < 0.2):
                break
                
            for features_max in feat_max_range:

                # this vectorizer will skip stop words
                vectorizer = CountVectorizer(
                    stop_words="english",
                    preprocessor=clean_text,
                    # Tuning for max_df and min_df
                    max_df = df_max,
                    min_df = df_min,
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

                #get p(y=0) and p(y=1) from training labels
                train_rows = imdb_labels.iloc[0:30000]
                train_rows = pd.Series(train_rows['sentiment'])
                train_rows_pos = train_rows.str.count("positive")
                train_rows_neg = train_rows.str.count("negative")
                num_pos = train_rows_pos.sum(axis = 0, skipna = True)
                num_neg = train_rows_neg.sum(axis = 0, skipna = True)
                p_neg = num_neg/30000
                p_pos = num_pos/30000
                

                #get total # of words and # of word i
                train_rows_pos = np.array([train_rows_pos])
                pos_word_v = np.dot(train_rows_pos, train_v)
                pos_total_words = pos_word_v.sum()
                pos_total_words += v_a


                train_rows_neg = np.array([train_rows_neg])
                neg_word_v = np.dot(train_rows_neg, train_v)
                neg_total_words = neg_word_v.sum()
                neg_total_words += v_a


                pos_word_v = (pos_word_v + a)
                neg_word_v = (neg_word_v + a)

                #Calculate Naive Bayes probability for seeing each word

                pos_P_word_v = np.divide(pos_word_v, pos_total_words)
                neg_P_word_v = np.divide(neg_word_v, neg_total_words)

                # p(y|x) validation
                pos_P_word_v = pos_P_word_v.transpose()
                pos_P_word_v = np.log(pos_P_word_v)
                pos_result_v = np.dot(valid_v, pos_P_word_v)
                pos_result_v = pos_result_v + np.log(p_pos)

                neg_P_word_v = neg_P_word_v.transpose()
                neg_P_word_v = np.log(neg_P_word_v)
                neg_result_v = np.dot(valid_v, neg_P_word_v)
                neg_result_v = neg_result_v + np.log(p_neg)
                

                accuracy = 0
                for i in range(10000):
                    if pos_result_v[i][0] > neg_result_v[i][0]:
                        if valid_labels[i][0] == "positive":
                            accuracy += 1
                    elif neg_result_v[i][0] > pos_result_v[i][0]:
                        if valid_labels[i][0] == "negative":
                            accuracy += 1

                accuracy = (accuracy/10000)*100
                #print("percentage accuracy with df_max=", df_max, " df_min=", df_min, "feature_max=", features_max, "is {}%".format(accuracy))

                if(accuracy > optAcc):
                    optMax = df_max
                    optMin = df_min
                    optFeat = features_max
                    optAcc = accuracy

                if(features_max == 3000):
                    feat_max_range = [optFeat]
                if(df_min_range == 0.5):
                    df_min_range = [optMin]
                if(df_max_range == 1.0):
                    df_max_range = [optMax]




    df_max          = optMax
    df_min          = optMin
    features_max    = optFeat

    #Testing data using optimal values

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        # Tuning for max_df and min_df
        max_df = df_max,
        min_df = df_min,
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

    #get p(y=0) and p(y=1) from training labels
    train_rows = imdb_labels.iloc[0:30000]
    train_rows = pd.Series(train_rows['sentiment'])
    train_rows_pos = train_rows.str.count("positive")
    train_rows_neg = train_rows.str.count("negative")
    num_pos = train_rows_pos.sum(axis = 0, skipna = True)
    num_neg = train_rows_neg.sum(axis = 0, skipna = True)
    p_neg = num_neg/30000
    p_pos = num_pos/30000
    

    #get total # of words and # of word i
    train_rows_pos = np.array([train_rows_pos])
    pos_word_v = np.dot(train_rows_pos, train_v)
    pos_total_words = pos_word_v.sum()
    pos_total_words += v_a


    train_rows_neg = np.array([train_rows_neg])
    neg_word_v = np.dot(train_rows_neg, train_v)
    neg_total_words = neg_word_v.sum()
    neg_total_words += v_a


    pos_word_v = (pos_word_v + a)
    neg_word_v = (neg_word_v + a)

    #Calculate Naive Bayes probability for seeing each word

    pos_P_word_v = np.divide(pos_word_v, pos_total_words)
    neg_P_word_v = np.divide(neg_word_v, neg_total_words)


    #get p(y=0) and p(y=1) from training labels
    train_rows = imdb_labels.iloc[0:30000]
    train_rows = pd.Series(train_rows['sentiment'])
    train_rows_pos = train_rows.str.count("positive")
    train_rows_neg = train_rows.str.count("negative")
    num_pos = train_rows_pos.sum(axis = 0, skipna = True)
    num_neg = train_rows_neg.sum(axis = 0, skipna = True)
    p_neg = num_neg/30000
    p_pos = num_pos/30000
    

    #get total # of words and # of word i
    train_rows_pos = np.array([train_rows_pos])
    pos_word_v = np.dot(train_rows_pos, train_v)
    pos_total_words = pos_word_v.sum()
    pos_total_words += v_a


    train_rows_neg = np.array([train_rows_neg])
    neg_word_v = np.dot(train_rows_neg, train_v)
    neg_total_words = neg_word_v.sum()
    neg_total_words += v_a

    pos_word_v = (pos_word_v + a)
    neg_word_v = (neg_word_v + a)

    #Calculate Naive Bayes probability for seeing each word

    pos_P_word_v = np.divide(pos_word_v, pos_total_words)
    neg_P_word_v = np.divide(neg_word_v, neg_total_words)


    # p(y|x) validation
    pos_P_word_v = pos_P_word_v.transpose()
    pos_P_word_v = np.log(pos_P_word_v)
    pos_result_v = np.dot(valid_v, pos_P_word_v)
    pos_result_v = pos_result_v + np.log(p_pos)

    neg_P_word_v = neg_P_word_v.transpose()
    neg_P_word_v = np.log(neg_P_word_v)
    neg_result_v = np.dot(valid_v, neg_P_word_v)
    neg_result_v = neg_result_v + np.log(p_neg)
    

    accuracy = 0
    for i in range(10000):
        if pos_result_v[i][0] > neg_result_v[i][0]:
            if valid_labels[i][0] == "positive":
                accuracy += 1
        elif neg_result_v[i][0] > pos_result_v[i][0]:
            if valid_labels[i][0] == "negative":
                accuracy += 1

    accuracy = (accuracy/10000)*100
    print("The optimal values are df_max=", df_max, ", df_min=", df_min, ", features_max=", features_max)
    print("the percentage accuracy is {}%".format(accuracy))


    # p(y|x) test
    pos_final_v = np.dot(test_v, pos_P_word_v)
    pos_final_v = pos_final_v + np.log(p_pos)

    neg_final_v = np.dot(test_v, neg_P_word_v)
    neg_final_v = neg_final_v + np.log(p_neg)

    
    field = ['sentiment']
    rows = []
    for i in range(10000):
        if pos_final_v[i][0] > neg_final_v[i][0]:
            rows.append(['positive'])
        elif neg_final_v[i][0] > pos_final_v[i][0]:
            rows.append(['negative'])

    with open('test-prediction3.csv', 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(field)  
        csvwriter.writerows(rows)
            
