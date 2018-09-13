### NAIVE BAYES CLASS
from nltk import word_tokenize
class naive_classifier:
    def _init_(self):
        self.trained = False

    def _train(self, corpus):
        print(corpus)
        #print ("Not implemented and I don't even care")
        
    def _read(self, document):
        toReturn = []
        with open(document) as f:
            for line in f.readlines():
                pair = line.split('\n')
                pair = pair[0].split('\t')
                bag = word_tokenize(pair[0].lower())
                label = int(pair[1])
                toReturn.append((bag,label))
        return toReturn
    
    def train(self, documents):
        """
        Takes txt inputs and trains the classifier
        """
        corpus = []
        for doc in documents:
            print ("reading: ",doc)
            for review in self._read(doc):
                corpus.append(review)
        #self._train(corpus)
        self.trained = True

if __name__ == '__main__':
    classifier = naive_classifier()
    print ("Ready to train")

