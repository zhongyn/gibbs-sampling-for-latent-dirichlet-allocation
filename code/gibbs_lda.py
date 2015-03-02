import numpy as np

class GibbsLDA(object):
    """Gibbs sampling for Latent Dirichlet Allocation."""

    def __init__(self, docword=None, vocab=None, alpha=1, beta=1, k=2):
        self.alpha = alpha
        self.beta = beta
        self.topics = k
        self.read_files(docword, vocab)
        self.init_tables()

    def read_files(self, docword, vocab):
        self.vocab = np.loadtxt(vocab, dtype='S')
        with open(docword) as f:
            self.docs_num = int(f.readline())
            self.vocab_num = int(f.readline())
            self.total_words = int(f.readline())
            self.docword = np.loadtxt(f,dtype=np.int)

    def init_tables(self):
        # axis: x->topicID, y->wordID
        self.topic_word_table = np.zeros((self.vocab_num, self.topics), dtype='uint16')
        # axis: x->topicID, y->docID
        self.topic_doc_table = np.zeros((self.docs_num, self.topics), dtype='uint16')
        self.expand_docword()

    def expand_docword(self):
        total_occurs = np.sum(self.docword[:,2])
        new_docword = np.zeros((total_occurs,), dtype=[('docID','uint16'),('wordID','uint16'),('topicID','uint8')])
        i = 0
        for w in self.docword:
            for j in range(w[2]):
                new_docword[i] = tuple(w)
                i += 1
        self.docword = new_docword
        # add topic column to docword array
        self.docword['topicID'] = np.random.randint(self.topics, size=self.docword.size)
        self.fill_tables()

    def fill_tables(self):
        # count the number of words in each topic for each word over all document
        self.docword = np.sort(self.docword, order=['wordID','topicID'])
        wordIDs, firstWordIDs, wordCounts = np.unique(self.docword['wordID'], return_index=True, return_counts=True)
        for i,valW in enumerate(firstWordIDs):
            topicIDs, topicCounts = np.unique(self.docword['topicID'][valW:valW+wordCounts[i]], return_counts=True)
            for j,val in enumerate(topicIDs):
                self.topic_word_table[wordIDs[i]-1,val] = topicCounts[j]

        # count the number of words in each topic for each document 
        self.docword = np.sort(self.docword, order=['docID','topicID'])
        docIDs, firstDocIDs, docCounts = np.unique(self.docword['docID'], return_index=True, return_counts=True)
        for i,valW in enumerate(firstDocIDs):
            topicIDs, topicCounts = np.unique(self.docword['topicID'][valW:valW+docCounts[i]], return_counts=True)
            for j,val in enumerate(topicIDs):
                self.topic_doc_table[docIDs[i]-1,val] = topicCounts[j]

    def gibbs_sampling(self):
        for i in range(10):
            print i
            if i > 0:
                break
            # j = 0
            for w in self.docword:
                # j += 1
                # print 
                # print j
                # if j>20:
                #     break
                # print w
                # print 'w[topic]:',w['topicID']
                # print self.topic_word_table[w['wordID']-1]
                # print self.topic_doc_table[w['docID']-1]
                self.update_topic_assignment(w)
                # print 'w[topic]:',w['topicID']
                # print self.topic_word_table[w['wordID']-1]
                # print self.topic_doc_table[w['docID']-1]


    def update_topic_assignment(self,w):
        prob = np.zeros(self.topics)
        wordID = w['wordID']-1
        docID = w['docID']-1
        topicID = w['topicID']

        self.topic_word_table[wordID,topicID] -= 1
        self.topic_doc_table[docID,topicID] -= 1

        for t in range(self.topics):
            left = (self.topic_word_table[wordID,t]+self.beta)*1.0/(np.sum(self.topic_word_table[:,t])+self.vocab_num*self.beta)
            right = (self.topic_doc_table[docID,t]+self.alpha)*1.0/(np.sum(self.topic_doc_table[docID])+self.topics*self.alpha)
            prob[t] = left*right
        prob = prob/np.sum(prob)
        print prob
        sample_topic = np.nonzero(np.random.multinomial(1,prob))[0][0]
        w['topicID'] = sample_topic
        self.topic_word_table[wordID,sample_topic] += 1
        self.topic_doc_table[docID,sample_topic] += 1


    def latent_variables(self):
        result = []
        for k in range(self.topics):
            top_wordIDs = np.argsort(self.topic_word_table[:,k])
            result.append(self.vocab[top_wordIDs[-20:]])
        self.result = np.array(result)

        # for k in range(lda.topics):
        #     result = np.sort(lda.topic_word_table[:,k])[-10:]
        #     print result
        #     words = []
        #     for j in result:
        #         words.append(lda.vocab[j-1])
        #     lda.result.append(words)
        #     print 'topic:',k
        #     print words




def read_test(args):
    lda = GibbsLDA(*args)
    # print lda.topic_doc_table
    # print lda.topic_word_table
    # print lda.docword[:50]
    # lda.gibbs_sampling()
    # lda.latent_variables()
    return lda

if __name__ == '__main__':
    nips = ['../data/docword.nips.txt', '../data/vocab.nips.txt',1,1,2]
    kos = ['../data/docword.kos.txt', '../data/vocab.kos.txt',1,1,2]
    nips_lda = read_test(nips)
    kos_lda = read_test(kos)
