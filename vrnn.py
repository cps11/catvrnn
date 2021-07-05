import torch
import torch.nn as nn
import torch.nn.functional as F

class VRNNCell(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VRNNCell, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())
        
        #recurrence
        self.rnn = nn.GRUCell(x_dim + z_dim, h_dim)

    def forward(self, x, h_prev):
        #encoder
        enc_t = self.enc(torch.cat([x, h_prev], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_std_t = self.enc_std(enc_t)

        #sampling and reparameterization
        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
    
        #decoder
        dec_t = self.dec(torch.cat([z_t, h_prev], 1))
        dec_mean_t = self.dec_mean(dec_t)
        
        #recurrence
        h = self.rnn(torch.cat([x, z_t], 1), h_prev)
        return  dec_mean_t, h

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        return eps.mul(std).add_(mean)


class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VRNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.h = None
        self.cell = VRNNCell(self.x_dim, self.h_dim, self.z_dim)
    
    def reset_hidden_state(self, h=None):
        self.h = h
    
    def forward(self, xs):
        N, T, D = xs.shape

        hs = torch.empty((N, T, self.h_dim), dtype=torch.float, device=xs.device)
        dec_means = torch.empty((N, T, self.x_dim), dtype=torch.float, device=xs.device)
        
        if self.h is None:
            self.h = torch.zeros((N, self.h_dim), dtype=torch.float, device=xs.device)
        
        for t in range(T):
            dec_mean_t, self.h = self.cell.forward(xs[:, t, :], self.h)
    
            hs[:, t, :] = self.h
            dec_means[:, t, :] = dec_mean_t
        
        return dec_means, hs


class VRNNlm(nn.Module):
    def __init__(self, vocab_size, k, x_dim, h_dim, z_dim, mode, init_h, device='cpu'):
        super(VRNNlm, self).__init__()
        self.h_dim = h_dim
        self.device = device
        self.mode = mode
        self.k = k

        self.embed = nn.Embedding(vocab_size, x_dim)
        self.vrnn  = VRNN(x_dim, h_dim, z_dim)
        self.fc4generate = nn.Linear(x_dim, vocab_size)
        self.fc4classfication = nn.Linear(h_dim, k)
        if init_h == 'static':
            self.init_h = self.init_static_h
        elif init_h == 'adaptive':
            self.h_init = nn.Linear(1, h_dim)
            self.init_h = self.init_adaptive_h

    def init_adaptive_h(self, labels, disturb=True):
        N = labels.size(0)
        labels = labels.float()
        h = self.h_init(labels.view(N, -1))
        if disturb:
            h += torch.rand([N, self.h_dim], device=labels.device)
        self.vrnn.reset_hidden_state(h)

    def init_static_h(self, labels, h_weight):
        data_size = labels.shape[0]
        h = torch.rand([data_size, self.h_dim], dtype=torch.float, device=labels.device)
        h = F.softmax(h, dim=-1)
        for i in labels.unique():
            idx = torch.where(i==labels)
            h[idx] = h[idx] * pow(-1, i) * h_weight

        self.vrnn.reset_hidden_state(h)

    def forward(self, xs, ts, labels, w):
        self.init_h(labels, w)
        xs = self.embed(xs)
        dec_means, hs = self.vrnn(xs)
        
        out = self.fc4generate(dec_means)
        N, T, D = out.shape
        gen_loss = F.cross_entropy(out.reshape(N * T, D), ts.reshape(N * T))
        if 'c' in self.mode:
            out = F.dropout(hs, 0.5)
            out = self.fc4classfication(hs[:, -1, :])
            cla_loss = F.cross_entropy(out, labels)
        else:
            cla_loss = torch.tensor(0)
        
        return gen_loss, cla_loss

    def predict(self, xs):
        xs = self.embed(xs)
        dec_means, hs = self.vrnn(xs)
        out = self.fc4generate(dec_means)
        out = F.softmax(out, dim=-1)

        return out, hs[:, -1, :]
    
    def sample(self, num_samples, labels, w, start_letter=0, sample_size=30):
        samples = torch.zeros([num_samples, sample_size], device=self.device, dtype=torch.long)
        inp = torch.full([num_samples, 1], start_letter, device=self.device, dtype=torch.long)

        self.init_h(labels, w)
        for t in range(sample_size):
            out, _ = self.predict(inp)
            out = torch.multinomial(out.squeeze(1), 1)
            samples[:, t] = out.view(-1).data
            inp = out

        return samples
