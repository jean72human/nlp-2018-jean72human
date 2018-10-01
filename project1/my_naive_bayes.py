
# coding: utf-8

# In[1]:


### NAIVE BAYES CLASS
from nltk import word_tokenize
import numpy as np
import re
class naive_classifier:
    def __init__(self, gram="unigram"):
        self.trained = False
        self.classes = ["positive", "negative"]
        self.nclasses = len(self.classes)
        
        self.gram = gram
        
        self.likelihoods = {c : dict() for c in range(self.nclasses) }
        self.priors = [0 for i in range(self.nclasses)]
        self.vocabulary = []

    def _train(self, corpus):
        classCounts = [0 for i in range(self.nclasses)]
        ndoc = len(corpus)
        wordCounts = {c : dict() for c in range(self.nclasses)}
        for document in corpus:
            review = document[0]
            label = document[-1]
            classCounts[label] += 1
            for word in review:
                if word in wordCounts[label].keys():
                    wordCounts[label][word] += 1
                else:
                    wordCounts[label][word] = 1
                    
        for index in range(len(self.classes)):
            self.priors[index] = np.log(classCounts[index]/ndoc)
            self.vocabulary += list(wordCounts[index].keys())
        self.vocabulary = set(self.vocabulary)
        print (len(self.vocabulary))
            
        for index in range(len(self.classes)):
            for word in self.vocabulary:
                if word in wordCounts[index]:
                    self.likelihoods[index][word] = np.log((wordCounts[index][word]+1)/(sum(wordCounts[index].values())+len(wordCounts[index])))
                else:
                    self.likelihoods[index][word] = np.log((1)/(sum(wordCounts[index].values())+len(wordCounts[index])))     
                    
            
        
        
    def _read(self, document):
        toReturn = []
        with open(document) as f:
            for line in f.readlines():
                pair = line.split('\n')
                pair = pair[0].split('\t')
                review = re.sub(r"[,?!-()*&^%|'.,]","",pair[0])
                bag = self.preprocess(pair[0].lower())
                label = int(pair[1])
                toReturn.append((self.biagramize(bag),label))
        return toReturn
    
    def preprocess(self, sentence):
        import string
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        words = word_tokenize(sentence)
        toReturn = []
        for word in words:
            if (word not in string.punctuation):
                toReturn.append(wordnet_lemmatizer.lemmatize(word))
        return toReturn
    
    def train(self, documents, test=False, split_ratio=0.3):
        """
        Takes txt inputs and trains the classifier
        """
        corpus = []
        for doc in documents:
            print ("reading: ",doc)
            for review in self._read(doc):
                corpus.append(review)
                
        if test:
            np.random.shuffle(corpus)
            split_point = int(len(corpus) * split_ratio)
            test_data = corpus[:split_point]
            train_data = corpus[split_point:]
            self._train(train_data)
            test_acc = self._test(test_data)
            train_acc = self._test(train_data)
            print (len(train_data)," training items")
            print (len(test_data)," testing items")
            print ("Training done")
            print ("Train accuracy: ",train_acc)
            print ("Test accuracy: ",test_acc)
        else:
            self._train(corpus)
            print ("Training done")
        self.trained = True
        
    def _predict(self, sentence):
        import operator
        """
        Takes tokenized input and outputs numerical class
        """
        sumc = dict()
        for c in range(self.nclasses):
            sumc[c] = self.priors[c]
            for word in sentence:
                if word in self.vocabulary:
                    sumc[c] += self.likelihoods[c][word]
        return max(sumc.items(), key=operator.itemgetter(1))[0]
    
    def predict(self, text):
        """
        Tokenize sentence, predicts and output class
        """
        sentence = self.biagramize(self.preprocess(text))
        return self._predict(sentence)
    
    def _test(self, data):
        n_items = len(data)
        n_correct = 0
        for document in data:
            review = document[0]
            label = document[-1]
            c = self._predict(review)
            if (c == label): n_correct += 1
        return n_correct / n_items
    
    def test_file(self, file_name):
        """
        Tests with a file and outputs a file of labels
        """
        labels = []
        with open(file_name) as f:
            for line in f.readlines():
                print(line,self.predict(line))
                labels.append(self.predict(line))
        
        with open('results_file.txt', 'w') as f:
            for label in labels:
                f.write(str(label)+"\n")
                
        print ("Results from ",file_name," printed to: output.txt")
        
    def test_accuracy(self, predicted, correct):
        """
        Takes a predicted output file and a correct labels file to calculate accuracy
        """
        
        predicted_labels = []
        correct_labels = []
        correct_predictions = 0
        
        with open(predicted) as f:
            for line in f.readlines():
                label = int(line)
                predicted_labels.append(label)
                
        with open(correct) as f:
            for line in f.readlines():
                label = int(line)
                correct_labels.append(label)
                
    def biagramize(self, words):
        """
        Turns unigrams into biagrams
        """
        if self.gram == "unigram":
            toReturn = words
        elif self.gram == "bigram":
            toReturn = []
            temp = ['<s>'] + words + ['</s>']
            for ind in range(len(temp)-1):
                toReturn.append(temp[ind]+'<>'+temp[ind+1])
        return toReturn
                
        
            
        
    
    def export(self, name):
        import json
        
        toExport = {
            "gram":self.gram,
            "likelihoods":self.likelihoods,
            "priors":self.priors,
            "vocabulary":self.vocabulary
        }
        
        np.save(name, toExport)
            
    def load(self, name):
        import json
        
        loaded = np.load(name)
            
        self.gram = loaded.item().get("gram")
        self.likelihoods = loaded.item().get("likelihoods")
        self.priors = loaded.item().get("priors")
        self.vocabulary = loaded.item().get("vocabulary")
            
            


if __name__ == '__main__':
    classifier = naive_classifier()
    try:
        print("Loading the model")
        classifier.load("model.npy")
    except:
        print ("Couldn't load the model. Training")

        classifier.train(["./sentiment_labelled_sentences/amazon_cells_labelled.txt",
                  "./sentiment_labelled_sentences/imdb_labelled.txt",
                  "./sentiment_labelled_sentences/yelp_labelled.txt"],
                 test=True,
                 split_ratio=0.2)
        
    name = input("Enter name of the file to test: ")
    print("Testing: ",name)
    try:
        classifier.test_file(name)
    except:
        print ("An error occured. Couldn't read the file")

