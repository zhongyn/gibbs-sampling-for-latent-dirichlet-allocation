import numpy as np

class GibbsLDA(object):
    """Gibbs sampling for Latent Dirichlet Allocation."""

    def __init__(self, docword=None, vocab=None, alpha=1, beta=1, k=5):
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
        self.topic_word_table = np.zeros((self.vocab_num, self.topics), dtype=np.int)
        self.topic_doc_table = np.zeros((self.docs_num, self.topics), dtype=np.int)
        # add topic column to docword array
        self.random_topic_assign = np.random.randint(self.topics, size=self.docword.shape[0])
        self.docword = np.column_stack((self.docword, self.random_topic_assign))
        self.docword = np.core.records.fromarrays(self.docword.transpose(),names='docID, wordID, count, topicID')

    def fill_tables(self):
        # count the number of words in each topic for each word over all document
        self.docword = np.sort(self.docword, order=['wordID','topicID'])
        _, firstWordIDs, wordCounts = np.unique(self.docword['wordID'], return_index=True, return_counts=True)
        for i,valW in enumerate(firstWordIDs):
            topicIDs, firstTopicIDs, topicCounts = np.unique(self.docword['topicID'][valW:valW+wordCounts[i]], return_index=True, return_counts=True)
            for j,valT in enumerate(firstTopicIDs):
                valT = valT + valW
                self.topic_word_table[i,topicIDs[j]] = np.sum(self.docword['count'][valT:valT+topicCounts[j]])

        # count the number of words in each topic for each document 
        self.docword = np.sort(self.docword, order=['docID','topicID'])
        _, firstDocIDs, docCounts = np.unique(self.docword['docID'], return_index=True, return_counts=True)
        for i,valW in enumerate(firstDocIDs):
            topicIDs, firstTopicIDs, topicCounts = np.unique(self.docword['topicID'][valW:valW+docCounts[i]], return_index=True, return_counts=True)
            for j,valT in enumerate(firstTopicIDs):
                valT = valT + valW
                self.topic_doc_table[i,topicIDs[j]] = np.sum(self.docword['count'][valT:valT+topicCounts[j]])




def read_test(file1, file2):
    lda = GibbsLDA(docword=file1, vocab=file2)
    lda.fill_tables()
    print lda.topic_word_table
    print lda.topic_doc_table
    return lda

if __name__ == '__main__':
    docword = '../data/docword.kos.txt'
    vocab = '../data/vocab.kos.txt'
    lda = read_test(docword, vocab)

