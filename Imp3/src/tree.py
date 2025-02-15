import numpy as np
import pandas as pd
import random

class Node():
    """
    Node of decision tree

    Parameters:
    -----------
    prediction: int
        Class prediction at this node
    feature: int
        Index of feature used for splitting on
    split: int
        Categorical value for the threshold to split on for the feature
    left_tree: Node
        Left subtree
    right_tree: Node
        Right subtree
    """
    def __init__(self, prediction, feature, split, left_tree, right_tree):
        self.prediction = prediction
        self.feature = feature
        self.split = split
        self.left_tree = left_tree
        self.right_tree = right_tree


class DecisionTreeClassifier():
    """
    Decision Tree Classifier. Class for building the decision tree and making predictions

    Parameters:
    ------------
    max_depth: int
        The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
    """

    def __init__(self, max_features=None, max_depth=None):
        self.max_depth = max_depth
        self.max_features = max_features

    # take in features X and labels y
    # build a tree
    def fit(self, X, y):
        self.num_classes = len(set(y))
        self.root = self.build_tree(X, y, depth=1)

    # make prediction for each example of features X
    def predict(self, X):
        preds = [self._predict(example) for example in X]

        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        node = self.root
        while node.left_tree:
            if example[node.feature] < node.split:
                node = node.left_tree
            else:
                node = node.right_tree
            return node.prediction

    # accuracy
    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum()/len(y)
        return accuracy

    # function to build a decision tree
    def build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on
        self.features_idx = np.arange(0, X.shape[1])

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None

        # what we would predict at this node if we had to
        # majority class
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        prediction = np.argmax(num_samples_per_class)

        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # create list of all features
            column_idx = list(self.features_idx)
            # Check if there is more features to use
            if self.max_features and self.max_features <= len(self.features_idx):
                column_idx = random.sample(population=column_idx, k=self.max_features)
            # consider each randomly selected feature
            for feature in column_idx:
                # consider the set of all values for that feature to split on
                possible_splits = np.unique(X[:, feature])
                for split in possible_splits:
                    # get the gain and the data on each side of the split
                    # >= split goes on right, < goes on left
                    gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
                    # if we have a better gain, use this split and keep track of data
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = split
                        best_left_X = left_X
                        best_right_X = right_X
                        best_left_y = left_y
                        best_right_y = right_y

        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
            right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
            return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


    # gets data corresponding to a split by using numpy indexing
    def check_split(self, X, y, feature, split):
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]

        # calculate gini impurity and gain for y, left_y, right_y
        gain = self.calculate_gini_gain(y, left_y, right_y)
        return gain, left_X, right_X, left_y, right_y

    def calculate_gini_gain(self, y, left_y, right_y):
        # not a leaf node
        # calculate gini impurity and gain
        gain = 0
        if len(left_y) > 0 and len(right_y) > 0:
            #gini index vals
            c_pos = y.sum()/len(y)
            cl_pos = left_y.sum()/len(left_y)
            cr_pos = right_y.sum()/len(right_y)

            #split benefit
            gain = self.calc_gini_u(c_pos, 1-c_pos) - self.calc_gini_branch(c_pos, 1-c_pos, cl_pos, 1-cl_pos) - self.calc_gini_branch(c_pos, 1-c_pos, cr_pos, 1-cr_pos)

            return gain
        # we hit leaf node
        # don't have any gain, and don't want to divide by 0
        else:
            return 0

    #gini index functions
    def calc_gini_u(self, c_pos, c_neg):
        return 1-(c_pos/(c_pos+c_neg))**2 - (c_neg/(c_pos+c_neg))**2

    def calc_gini_branch(self, c_pos, c_neg, c_branch_pos, c_branch_neg):
        p_branch = (c_branch_pos+c_branch_neg)/(c_pos+c_neg)
        u_branch = self.calc_gini_u(c_branch_pos, c_branch_neg)
        return p_branch*u_branch

class RandomForestClassifier():
    """
    Random Forest Classifier. Build a forest of decision trees.
    Use this forest for ensemble predictions

    YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

    Parameters:
    -----------
    n_trees: int
        Number of trees in forest/ensemble
    max_features: int
        Maximum number of features to consider for a split when feature bagging
    max_depth: int
        Maximum depth of any decision tree in forest/ensemble
    """
    def __init__(self, n_trees, max_features, max_depth):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth

        ##################
        self.forest = []
        random.seed(0)
        ##################

    # fit all trees
    def fit(self, X, y):
        bagged_X, bagged_y = self.bag_data(X, y)
        print('Fitting Random Forest...\n')
        for i in range(self.n_trees):
            print(i+1, end='\t\r')
            ##################
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(bagged_X, bagged_y)
            self.forest.append(tree)
            ##################
        print()

    def bag_data(self, X, y, proportion=1.0):
        bagged_X = []
        bagged_y = []
        for i in range(self.n_trees):
            #continue
            ##################
            j = np.random.randint(low=0, high=len(X))
            bagged_X.append(X[j])
            bagged_y.append(y[j])
            # print('adding {}'.format(j))
            ##################

        # ensure data is still numpy arrays
        return np.array(bagged_X), np.array(bagged_y)


    def predict(self, X):
        preds = []

        # remove this one \/
        #preds = np.ones(len(X)).astype(int)
        # ^that line is only here so the code runs

        ##################
        pred = {}
        for i in range(len(self.forest)):
            col_name = "tree_{}".format(i)
            predictions = self.forest[i].predict(X)
            pred[col_name] = predictions
        pred = pd.DataFrame(pred)
        preds = pred.mode(axis=1)[0]
        ##################
        return preds


################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################
class AdaBoostClassifier():
    """
    AdaBoost Classifier, create weighted classifiers
    Use this calssifier to predict the results

    Parameters:
    -----------
    sample_weights: double?
        The weights for the data
    stumps: trees with depth of 1
        The trees each with a depth of 1
    stump_weights: vector of weights
        The weights for each tree
    errors: vector of errors for each tree
        The errors in classification for each stump
    """
    def __init__(self):
        self.sample_weights = None
        self.stumps = None
        self.stump_weights = None
        self.errors = None

    def fit(self, X, y):
        pass


    def predict(self, X):
        preds = []

        # remove this one \/
        preds = np.ones(len(X)).astype(int)

        return preds




