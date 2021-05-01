
from nltk.tokenize import RegexpTokenizer, word_tokenize
from collections import defaultdict
import json
import math
import numpy as np


class vector_space_model:
    def __init__(self):
        self.raw_text = ""
        self.files = [[]]
        self.processed_text = []
        self.index = {}
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.doc_vectors = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10:[],
                            11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[],
                            21:[], 22:[], 23:[], 24:[], 25:[], 26:[], 27:[], 28:[], 29:[], 30:[],
                            31:[], 32:[], 33:[], 34:[], 35:[], 36:[], 37:[], 38:[], 39:[], 40:[],
                            41:[], 42:[], 43:[], 44:[], 45:[], 46:[], 47:[], 48:[], 49:[], 50:[]
                            }
        self.query = []


    
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

    def tf_idf(self, query):
        #form partial doc vectors
        # print( self.index)
        self.query = list( query.split(" ") )
        for term in self.query:
            df = self.index[term][0]
            idf = math.log( 50/df , 10)

            for i in range(1,51):
                docs = list( self.index[term][1] )
                if i in docs:
                    tfidf = self.index[term][1][i] * idf
                    self.doc_vectors[i].append(tfidf)
                else:
                    self.doc_vectors[i].append(0)
        
        return self.query, self.doc_vectors
    
    def cosine_sim(self):
        ranks = {}
        query_mag = math.sqrt( 1 * len(self.query) ) 
        query_vec = [1 * len(self.query) ]
        #considering query vector as a vector of 1s since all query words are assumed to be present in the dictionary
        for i in range(1,51):
            # self.doc_vectors[i]
            doc = self.doc_vectors[i]
            doc_mag = sum( np.square(doc) )
            if doc_mag > 0:
                cross_prod = np.multiply( query_vec, doc ).sum()

                score = cross_prod / (doc_mag * query_mag) 
            else:
                score = 0
        
            ranks[i] = score

        result = sorted(ranks.items(), key=lambda item: item[1] , reverse=False)
        # print(*ranks, sep='\n')
        result = [x[0] for x in result if x[1] >= 0.05]

        # return result
        # print(*result, sep='\n')

        titles = []
        for ids in result:
            with open( 'ShortStories' + "/" + str(ids) + ".txt","r") as file:
                titles.append( [str(ids),  file.readline() ] )
        
        return titles




                


# x = vector_space_model()
# x.get_files('ShortStories')
# x.get_raw_text('ShortStories')
# x.preprocess()
# x.set_index() 
# x.store_index()
# print(x.tf_idf('front of the lodge faces the hospital'))
# print(*x.cosine_sim(), sep="\n")
