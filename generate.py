import torch
from vrnn import VRNNlm
import dataset
import os
import argparse

def breakmodelname(modelpath):
    name = os.path.split(modelpath)[-1]
    name = os.path.splitext(name)[0]
    corpus, mode, tag, w, ts = name.split('_')
    if w != 'True':
        w = float(w)
    else:
        w = False
    return mode, corpus, tag, w, ts

def outpath(mode, corpus, tag, w, ts, label):
    root = os.path.split(os.path.realpath(__file__))[0]
    file = '{}_{}_{}_{}_{}_{}.txt'.format(corpus, mode, tag, w, ts, label)

    return os.path.join(root, 'txt', file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='generate text by a catvrnn model')
    parser.add_argument('-d', type=int, default=0, help='the id of gpu')
    parser.add_argument('-b', type=int, default=32, help='the size of batch. default is 32')
    parser.add_argument('-e', type=int, default=300, help='the dimension of embeding layer\'s output . default is 300')
    parser.add_argument('-hs', type=int, default=256, help='the dimension of RNN\'s hidden state. default is 356')
    parser.add_argument('-z', type=int, default=128, help='the dimension of VAE\'s latent vector. default is 128')
    parser.add_argument('-w', type=int, default=None, help='the weight to initial RNN\'s hidden state.')
    parser.add_argument('-l', type=int, default=None, help='the label of generated texts. default is no goal to generate special label texts.')
    parser.add_argument('-c', type=int, default=5000, help='the count of generated texts')
    parser.add_argument('-s', type=int, default=30, help='the length of generated texts')
    parser.add_argument('-m', type=str, help='a filename of the model to generate texts', required=True)
    args = parser.parse_args()
    
    batch_size = args.b
    x_dim = args.e
    h_dim = args.hs
    z_dim = args.z
    label = args.l
    num_samples = args.c
  
    mode, corpus, tag, h_weight, ts = breakmodelname(args.m)
    if args.w:
        h_weight = args.w

    print('w={}'.format(h_weight))
    _, labels, word_to_id, id_to_word = getattr(dataset, corpus)().load_data()
    vocab_size = len(word_to_id)
    k = len(set(labels))

    device = torch.device("cuda:{}".format(args.d) if torch.cuda.is_available() else "cpu")
    model = VRNNlm(vocab_size, k, x_dim, h_dim, z_dim, 'g', tag, device)
    model.load_state_dict(torch.load(args.m))
    model.to(device)
    model.eval()

    labels = []
    if label is None:
        labels = [i for i in range(k)]
    else:
        labels = [label]

    for label in labels:
        out = outpath(mode, corpus, tag, h_weight, ts, label)
        docs = ''
        count = 0 
        while count < num_samples:
            next_count = num_samples - count
            if next_count >= batch_size:
                next_count = batch_size
            count += next_count
            
            labels = torch.full([next_count], label, dtype=torch.long, device=device)
            samples = model.sample(next_count, labels, h_weight, 0, args.s)
            for doc in samples.tolist():
                docs += ' '.join(id_to_word[i] for i in doc) + '\n'
        
        with open(out, 'w') as f:
            f.write(docs)
        print(out)