import os
import sys
import pickle
import random

class Br(object):
    def __init__(self):
        vocab_file = 'br.vocab.pkl'
        name = 'br'
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.current_dir, name)
        self.vocab_path = os.path.join(self.dataset_dir, vocab_file)
        self.files = {0: 'pos.txt', 1: 'neg.txt'}
    
    def readfile(self, path, label):
        corpus = []
        labels = []
        with open(path, 'rb') as fd:
            for line in fd.readlines():
                words = ['<go>'] + [w for w in line.decode(errors='ignore').strip().split(' ')] + ['<cls>']
                l = len(words)
                if l < 15 + 2 or l > 30 + 2:
                    continue

                corpus.append(words)
                labels.append(label)

        random.shuffle(corpus)
        return corpus[:6000], labels[:6000]
    
    def count_stop_words(self, corpus, min_count):
        word_count = {}
        for words in corpus:
            for w in words:
                if w in word_count:
                    word_count[w] += 1
                else:
                    word_count[w] = 1
        
        stop_words = set()
        for w, count in word_count.items():
            if count <= min_count:
                stop_words.add(w)
        
        return stop_words

    def load_vocab(self):
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'rb') as f:
                word_to_id, id_to_word = pickle.load(f)
            return word_to_id, id_to_word
        
        self.corpus = []
        self.lables = []
        for label, filename in self.files.items():
            file_path = os.path.join(self.dataset_dir, filename)
            docs, labels = self.readfile(file_path, label)

            self.corpus.extend(docs)
            self.lables.extend(labels)
        
        stop_word = self.count_stop_words(self.corpus, 0)
        print('stop_word', len(stop_word))
        word_to_id = {}
        id_to_word = {}
        for words in self.corpus:
            for w in words:
                if w in stop_word:
                    w = '<unk>'

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
        
        # corpus = []
        # labels = []
        # max_length = 0
        # for label, filename in self.files.items():
        #     file_path = os.path.join(self.dataset_dir, filename)
        #     docs, tags, mlen = self.readfile(file_path, label)

        #     corpus.extend(docs)
        #     labels.extend(tags)
        #     max_length = (mlen > max_length) and mlen or max_length

        idlists = []
        cls_id = word_to_id['<cls>']
        for doc in self.corpus:
            # words = [word_to_id.get(w, word_to_id['<unk>']) for w in doc]
            words = [word_to_id[w] for w in doc]
            words += [cls_id] * (32 - len(words))
            idlists.append(words)

        with open(save_path, 'wb') as f:
            pickle.dump((idlists, self.lables), f)

        print(len(word_to_id))
        return idlists, self.lables, word_to_id, id_to_word
        
        
if __name__ == '__main__':
    data = Br()
    _, labels, word_to_id, _ = data.load_data()
    import numpy
    labels = numpy.array(labels)
    print(numpy.unique(labels, return_counts=True))
    print(len(word_to_id))
