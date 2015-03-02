import numpy as np
import time

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

    # def gibbs_sampling(self):
    #     for i in range(10):
    #         print i
    #         if i > 10:
    #             break
    #         j = 0
    #         print 'docword.size:',self.docword.size
    #         for w in self.docword:
    #             j += 1
    #             if j%10000 == 0:
    #                 print j
                # self.update_topic_assignment(w)

    def gibbs_sampling(self):
        prob = np.zeros(self.topics)
        self.words_result = []
        # self.prob_result = []
        for i in range(30):
            print i
            # t = time.time()
            # if i > 3:
            #     break
            # j = 0
            print 'docword.size:',self.docword.size
            for w in self.docword:
                # j += 1
                # if j == 10000:
                #     break
                wID = w['wordID']-1
                dID = w['docID']-1
                tID = w['topicID']
                self.topic_word_table[wID,tID] -= 1
                self.topic_doc_table[dID,tID] -= 1

                for t in range(self.topics):
                    left = (self.topic_word_table[wID,t]+self.beta)*1.0/(np.sum(self.topic_word_table[:,t])+self.vocab_num*self.beta)
                    right = (self.topic_doc_table[dID,t]+self.alpha)*1.0/(np.sum(self.topic_doc_table[dID])+self.topics*self.alpha)
                    prob[t] = left*right
                prob = prob/np.sum(prob)
                # print prob
                sample_topic = np.nonzero(np.random.multinomial(1,prob))[0][0]
                w['topicID'] = sample_topic
                self.topic_word_table[wID,sample_topic] += 1
                self.topic_doc_table[dID,sample_topic] += 1
            self.latent_variables()
            # print time.time()-t

    # def update_topic_assignment(self,w):
        # prob = np.zeros(self.topics)
        # wordID = w['wordID']-1
        # docID = w['docID']-1
        # topicID = w['topicID']

        # self.topic_word_table[wordID,topicID] -= 1
        # self.topic_doc_table[docID,topicID] -= 1

        # for t in range(self.topics):
        #     left = (self.topic_word_table[wordID,t]+self.beta)*1.0/(np.sum(self.topic_word_table[:,t])+self.vocab_num*self.beta)
        #     right = (self.topic_doc_table[docID,t]+self.alpha)*1.0/(np.sum(self.topic_doc_table[docID])+self.topics*self.alpha)
        #     prob[t] = left*right
        # prob = prob/np.sum(prob)
        # # print prob
        # sample_topic = np.nonzero(np.random.multinomial(1,prob))[0][0]
        # w['topicID'] = sample_topic
        # self.topic_word_table[wordID,sample_topic] += 1
        # self.topic_doc_table[docID,sample_topic] += 1


    def latent_variables(self):
        top_words = []
        top_probs = []
        for k in range(self.topics):
            top_wordIDs = np.argsort(self.topic_word_table[:,k])
            top_words.append(self.vocab[top_wordIDs[-10:]])
            top_probs.append(self.topic_word_table[top_wordIDs[-10:],k]*1.0/np.sum(self.topic_word_table[:,k]))
        self.top_words = np.array(top_words)
        self.top_probs = np.array(top_probs)
        print self.top_words
        print self.top_probs
        self.words_result.append(top_words)
        # self.prob_result.append(top_probs)

    def save_result(self):
        print self.words_result
        # print self.prob_result
        np.savetxt('../data/top_words_'+str(self.topics)+'.txt', self.words_result, delimiter=',', fmt='%s')
        np.savetxt('../data/top_probs_'+str(self.topics)+'.txt', self.top_probs, delimiter=',', fmt='%f')



def read_test(args):
    lda = GibbsLDA(*args)
    # lda.gibbs_sampling()
    # lda.latent_variables()
    return lda

def main(args):
    topics = [2,5,10]
    alpha = 1
    beta = 1
    for t in topics:
        new_args = args+[alpha,beta,t]
        lda = GibbsLDA(*new_args)
        lda.gibbs_sampling()
        lda.save_result()


if __name__ == '__main__':
    nips_input = ['../data/docword.nips.txt', '../data/vocab.nips.txt']
    kos_input = ['../data/docword.kos.txt', '../data/vocab.kos.txt']
    # nips = read_test(nips_input)
    # kos = read_test(kos_input)
    main(kos_input)
