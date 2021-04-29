
from nltk.tokenize import RegexpTokenizer, word_tokenize
from collections import defaultdict
import json


class postings:
    def __init__(self):
        self.raw_text = ""
        self.files = [[]]
        self.processed_text = []
        self.index = {}
        self.tokenizer = RegexpTokenizer(r'\w+')


    
    # get raw text from files
    def get_raw_text(self, file):
        for i in range(1,51):
        # ss = open( "ShortStories/" + str(i) + ".txt","r") 
            ss = open( file + "/" + str(i) + ".txt","r") 
            for text in ss:
                self.raw_text = self.raw_text + text.strip() + " "
        ss.close()
        return
    
    def preprocess(self):
        ## preprocess to get tokens
        # tokenizer = RegexpTokenizer(r'\w+')
        self.raw_text = self.tokenizer.tokenize(self.raw_text)
        # case fold to lowercase
        self.processed_text = [w.lower() for w in self.raw_text]
        # stem and order alphabetically
        self.processed_text = list( sorted( set( self.processed_text )))
        return self.processed_text
    
    def get_files(self, file):
        for j in range(1,51):
            text = ""
            f = open( file + "/" + str(j) + ".txt","r")
            for lines in f:
                text = text + lines.strip() + " "
            f.close()
            docid_tokens = self.tokenizer.tokenize(text)
            # case fold to lowercase
            docid_words = [w.lower() for w in docid_tokens]
            # print(docid_words)
            self.files.append(docid_words)

    def set_index(self):
        for word in self.processed_text:
            df = 0
            temp= {}
            for i in range(51):
                if word in self.files[i]:
                    df += 1
                    tf_i =len( [i for i, x in enumerate(self.files[i]) if x == word] )
                    temp[i] = tf_i
            self.index[word] = [df, temp] 
        
        return self.index

    def store_index(self):
        with open('index.txt', 'w') as file:
            file.write(json.dumps(self.index)) # use `json.loads` to do the reverse
        file.close()

    def tf_idf(query):
        #form partial doc vectors
        for term in query:
            df = self.index[term][0]
            idf = math.log( df , 10)

            for i in range(df):
                


x = postings()
x.get_files('ShortStories')
x.get_raw_text('ShortStories')
x.preprocess()
print( x.set_index() )
x.store_index()