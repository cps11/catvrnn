import torch.optim as optim
import torch
from vrnn import VRNNlm
import dataset
from tqdm import tqdm
import pickle
import argparse
import datetime
import math
import os

def outpath(name, mode, w, tag):
    root = os.path.split(os.path.realpath(__file__))[0]
    now = datetime.datetime.now().strftime('%m%d')
    file = '{}_{}_{}_{}_{}'.format(name, mode, tag, w, now)

    modepath = os.path.join(root, 'pth', file) + '.pth'
    ckppath = os.path.join(root, 'checkpoint', file) + '.pth'
    losspth = os.path.join(root, 'loss', file) + '.pkl'

    return modepath, ckppath, losspth
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train a vrnn to generate text.')
    parser.add_argument('-d', type=int, default=0, help='the id of gpu')
    parser.add_argument('-b', type=int, default=32, help='the size of batch. default is 32')
    parser.add_argument('-e', type=int, default=300, help='the dimension of embeding layer. default is 300')
    parser.add_argument('-hs', type=int, default=256, help='the dimension of RNN\'s hidden state. default is 356')
    parser.add_argument('-w', type=float, default=None, help='the weight to initial RNN\'s hidden state. default is 0')
    parser.add_argument('-z', type=int, default=128, help='the dimension of VAE\'s latent vector. default is 128')
    parser.add_argument('-lr', type=float, default=0.001, help='lr. default is 0.001')
    parser.add_argument('-me', type=int, default=250, help='the max epoch. default is 250')
    parser.add_argument('-tm', type=str, default='gc', choices=['g', 'gc'], help='the training mode. g is gererate; gc is gererate classification at same time. default is gc')
    parser.add_argument('-n', type=str, help='the name of corpus', required=True)
    parser.add_argument('-f', type=str, help='static or adaptive', choices=['static', 'adaptive'], required=True)
    args = parser.parse_args()
    
    batch_size = args.b
    x_dim = args.e
    h_dim = args.hs
    if args.f == 'static' and not isinstance(args.w, float):
        raise ValueError('-w must be a float when -f is static. it is {}'.format(args.w))
    elif args.f == 'adaptive' and args.w is not None:
        raise ValueError('-w must not be set when -f is adaptive. it is {}'.format(args.w))
    h_weight = (args.w is None) and True or args.w
    z_dim = args.z
    lr = args.lr
    max_epoch = args.me
    gid = args.d

    modepath, ckppath, losspth = outpath(args.n, args.tm, h_weight, args.f)

    train_corpus, train_labels, word_to_id, id_to_word = getattr(dataset, args.n)().load_data()
    train_corpus = torch.tensor(train_corpus)
    train_labels = torch.tensor(train_labels)

    vocab_size = len(word_to_id)
    data_size = train_corpus.shape[0]
    max_iters = math.ceil(data_size / batch_size)
    k = torch.unique(train_labels).shape[0]

    print('batch size:', batch_size)
    print('embeding size:', x_dim)
    print('hidden state size:', h_dim)
    print('latent size:', z_dim)
    print('weight to initialize the hidden state:', h_weight)
    print('method of initialize the hidden state:', args.f)
    print('lr:', lr)
    print('max epoch:', max_epoch)
    print('dataset:', args.n)
    print('vocab size:', vocab_size)
    print('data size:', data_size)
    print('traing mode:', args.tm)
    print('k of classfication:', k)
    print('gpu id:', gid)
    print('model path:', modepath)
    print('checkpoint path:', ckppath)
    print('loss path:', losspth)
    
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:{}".format(gid) if torch.cuda.is_available() else "cpu")
    model = VRNNlm(vocab_size, k, x_dim, h_dim, z_dim, args.tm, args.f, device)
    model.to(device)
    model.train()
 
    train_corpus = train_corpus.to(device)
    train_labels = train_labels.to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=lr)
    record = []
    minloss = 1000000
    for epoch in range(max_epoch):
        loss_count = 0
        total_loss = 0
        genloss = 0
        claloss = 0

        idx = torch.randperm(data_size)
        xs = train_corpus[idx, :-1]
        ts = train_corpus[idx, 1:] 
        labels = train_labels[idx]
        
        pbar = tqdm(range(max_iters))
        for i in pbar:
            pbar.set_description('Processing {}/{}'.format(epoch, max_epoch))
            offset = i * batch_size

            optimizer.zero_grad()
            gen_loss, cla_loss = model(xs[offset:offset+batch_size], ts[offset:offset+batch_size], labels[offset:offset+batch_size], h_weight)
            loss = gen_loss + cla_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            loss_count += 1
            total_loss += loss.item()
            genloss += gen_loss.item()
            claloss += cla_loss.item()

            pbar.set_postfix(total_loss=loss.item(), gen_loss=gen_loss.item(), cla_loss=cla_loss.item())

        total_loss /= loss_count
        genloss /= loss_count
        claloss /= loss_count
        pbar.set_postfix(total_loss=total_loss, gen_loss=genloss, cla_loss=claloss)
            
        if total_loss < minloss:
            minloss = total_loss
            torch.save(model.state_dict(), ckppath)
        record.append([total_loss, gen_loss, cla_loss])
    
    torch.save(model.state_dict(), modepath)
    with open(losspth, 'wb') as f:
        pickle.dump(record, f)
    
    print('model path:', modepath)
    print('checkpoint path:', ckppath)
    print('loss path:', losspth)
    
