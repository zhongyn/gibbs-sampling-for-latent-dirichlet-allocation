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
        self.assign_word_topic = np.random.randint(self.topics, size=self.docword.shape[0])

    def assign():
        pass










def read_test(file1, file2):
    lda = GibbsLDA(docword=file1, vocab=file2)
    print lda.vocab
    print lda.docword
    return lda

if __name__ == '__main__':
    docword = '../data/docword.kos.txt'
    vocab = '../data/vocab.kos.txt'
    lda = read_test(docword, vocab)

