
# coding: utf-8



## importing all the libraries needed

from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import nltk
import numpy as np
import pandas as pd
import re
from io import StringIO

import sys



class classifier:
    
    def __init__(self, normalize=True, clf_type = "logistic", split_ratio=0.3):
        """
        Initializes the class with the right classifier attribute depending on the type of classifier
        """
        if clf_type == "logistic":
            self.clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
        elif clf_type == "naive":
            self.clf = naive_bayes.MultinomialNB()
        self.normalize = normalize
        if self.normalize:
            self.vec = TfidfVectorizer(use_idf=True)
        else:
            self.vec = TfidfVectorizer(use_idf=True, lowercase = True, strip_accents=ascii, stop_words = set(nltk.corpus.stopwords.words('english')))

        
    def _read(self, documents):
        """
        Reads and combines all the documents in one big pandas data frame
        """
        data = []
        X,Y = [], []
        for document in documents:
            d_ata = pd.read_csv(document, sep='\t', names=['review','label'])
            data.append(d_ata)
        data = pd.concat(data)
        self.data = data
        Y = data.label
        self.vec.fit(data.review)
        X = self.preprocess(data)
        
        return train_test_split(X,Y)
    
    def preprocess(self, data_f):
        """
        Preprocesses the text data by turning it into frequency tables
        Does a few normalization steps (lowercasing, removing stopwords ...) if self.normalize = true
        """
        
        return self.vec.transform(data_f.review)
    
    def train(self, documents):
        """
        Calls the train function
        Trains the classifier object
        """
        X_train, X_test, Y_train, Y_test =  self._read(documents)       
                
        self.clf.fit(X_train,Y_train)
        
        acc = roc_auc_score(Y_test,self.clf.predict_proba(X_test)[:,1])
        
        print ("Accuracy: ",acc)
        
    def predict(self, sentence):
        """
        Predicts for a sentence
        """
        data = pd.read_csv(StringIO(sentence), names=['review'])
        X = self.preprocess(data)
        Y = self.clf.predict_proba(X)
        
        return np.argmax(Y)
    
    def test_file(self, file_name, version, classifier_type):
        """
        Tests with a file and outputs a file of labels
        """
        labels = []
        with open(file_name) as f:
            for line in f.readlines():
                print(line,self.predict(line))
                labels.append(self.predict(line))
                
        filename = 'test_results-' + classifier_type + '-' + version + '.txt'
        
        with open(filename, 'w') as f:
            for label in labels:
                f.write(str(label)+"\n")
                
        print ("Results from ",file_name," printed to:",filename)
                

if __name__ == '__main__':

    version, classifier_type, test_file = sys.argv[2], sys.argv[1], sys.argv[3]


    if (classifier_type == 'lr'):
        if (version == 'u'):
            print ("Unnormalized data, Logistic regression")
            my_clf = classifier(normalize=False)
            my_clf.train(["../project1/sentiment_labelled_sentences/amazon_cells_labelled.txt",
                              "../project1/sentiment_labelled_sentences/imdb_labelled.txt",
                              "../project1/sentiment_labelled_sentences/yelp_labelled.txt"])
        elif (version == 'n'):
            print ("Normalized data, Logistic regression")
            my_clf = classifier(normalize=True)
            my_clf.train(["../project1/sentiment_labelled_sentences/amazon_cells_labelled.txt",
                              "../project1/sentiment_labelled_sentences/imdb_labelled.txt",
                              "../project1/sentiment_labelled_sentences/yelp_labelled.txt"])
    elif (classifier_type == 'nb'):
        if (version == 'u'):
            print ("Unnormalized data, Naive Bayes")
            my_clf = classifier(normalize=False, clf_type='naive')
            my_clf.train(["../project1/sentiment_labelled_sentences/amazon_cells_labelled.txt",
                              "../project1/sentiment_labelled_sentences/imdb_labelled.txt",
                              "../project1/sentiment_labelled_sentences/yelp_labelled.txt"])
        elif (version == 'n'):
            print ("Normalized data, Naive Bayes")
            my_clf = classifier(normalize=True, clf_type='naive')
            my_clf.train(["../project1/sentiment_labelled_sentences/amazon_cells_labelled.txt",
                              "../project1/sentiment_labelled_sentences/imdb_labelled.txt",
                              "../project1/sentiment_labelled_sentences/yelp_labelled.txt"])



    print()
    print()
    
    my_clf.test_file(test_file, version, classifier_type)

