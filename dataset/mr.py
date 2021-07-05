import os
import pickle

class Mr(object):
    def __init__(self):
        vocab_file = 'mr.vocab.pkl'
        name = 'mr'
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.current_dir, name)
        self.vocab_path = os.path.join(self.dataset_dir, vocab_file)
        self.files = {0: 'rt-polarity.pos', 1: 'rt-polarity.neg'}
    
    def readfile(self, path, label):
        corpus = []
        labels = []
        max_length = 0
        with open(path, 'rb') as fd:
            for line in fd.readlines():
                words = ['<go>'] + [w for w in line.decode(errors='ignore').strip().split(' ')] + ['<cls>']
                l = len(words)
                if l < 15 + 2 or l > 30 + 2:
                    continue

                if l > max_length:
                    max_length = l
            
                corpus.append(words)
                labels.append(label)
        
        return corpus, labels, max_length
    
    def load_vocab(self):
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'rb') as f:
                word_to_id, id_to_word = pickle.load(f)
            return word_to_id, id_to_word
        
        corpus = []
        for label, filename in self.files.items():
            file_path = os.path.join(self.dataset_dir, filename)
            docs, _, _ = self.readfile(file_path, label)

            corpus.extend(docs)
        
        word_to_id = {}
        id_to_word = {}
        for words in corpus:
            for w in words:
                if w in word_to_id:
                    continue
                    
                tmp_id = len(word_to_id)
                word_to_id[w] = tmp_id
                id_to_word[tmp_id] = w
                
        with open(self.vocab_path, 'wb') as f:
            pickle.dump((word_to_id, id_to_word), f)

        return word_to_id, id_to_word
    
    
    def load_data(self):
        save_path = os.path.join(self.dataset_dir, 'corpus.pkl')
        word_to_id, id_to_word = self.load_vocab()

        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                corpus, labels = pickle.load(f)
            return corpus, labels, word_to_id, id_to_word
        
        corpus = []
        labels = []
        max_length = 0
        for label, filename in self.files.items():
            file_path = os.path.join(self.dataset_dir, filename)
            docs, tags, mlen = self.readfile(file_path, label)

            corpus.extend(docs)
            labels.extend(tags)
            max_length = (mlen > max_length) and mlen or max_length

        idlists = []
        cls_id = word_to_id['<cls>']
        for doc in corpus:
            words = [word_to_id[w] for w in doc]
            words += [cls_id] * (max_length - len(words))
            idlists.append(words)

        with open(save_path, 'wb') as f:
            pickle.dump((idlists, labels), f)

        # print(len(word_to_id))
        return idlists, labels, word_to_id, id_to_word
        
        
if __name__ == '__main__':
    data = Mr()
    _, labels, word_to_id, _ = data.load_data()
    import numpy
    labels = numpy.array(labels)
    print(numpy.unique(labels, return_counts=True))
    print(len(word_to_id))
