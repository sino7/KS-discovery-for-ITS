import numpy as np
import torch

import pickle as pk

from utils import masked_softmin

class SKTCell(torch.nn.Module):

    def __init__(self, n_ex, n_kc, d_e, d_c, d_h, ex_graph):
        super(SKTCell, self).__init__()

        self.n_ex = n_ex
        self.n_kc = n_kc
        self.d_e = d_e
        self.d_c = d_c
        self.d_h = d_h

        self.ex_graph = ex_graph
        
        # KS
        self.kc_gamma = torch.nn.Parameter(torch.randn(n_kc, n_kc))

        # Embeddings
        self.ex_embeddings = torch.nn.Linear(2*n_ex, d_e, bias=False)
        self.kc_embeddings = torch.nn.Linear(n_kc, d_c, bias=False)
        
        # Temporal propagation
        self.tgru = torch.nn.GRUCell(d_e, d_h)
        
        # Spatial directed propagation
        # Here we only model the prerequisite relations
        self.fpart = torch.nn.Linear(d_c+d_h, d_h)
        
        # Final update
        self.ulinear = torch.nn.Linear(d_h, d_h)
        self.ugru = torch.nn.GRUCell(d_h, d_h)


    def forward(self, h, ex, su):
        # Update the recurrent state h based on x
        # The recurrent state is factored according to each KC
        # Input:
        #   h: Tensor of shape (batch_size, n_kc, d_h) - structured hidden state
        #   ex: Tensor of shape (batch_size, n_ex) - taken exercise
        #   su: Tensor of shape (batch_size) - response on the exercise
        # Output:
        #   h: Tensor of shape (batch_size, n_kc, d_h) - updated hidden state
        batch_size = ex.shape[0]
        
        # KC graph
        kc_graph = torch.sigmoid(self.kc_gamma)

        # Exercise embeddings
        x_id = torch.zeros(batch_size, 2, self.n_ex)
        x_id = ex.unsqueeze(1).repeat(1, 2, 1)  # (batch_size, 2, n_ex)
        x_id[:, 0] = ex * su.unsqueeze(1).repeat(1, self.n_ex)
        x_id[:, 1] = ex * (1-su).unsqueeze(1).repeat(1, self.n_ex)
        x_id = x_id.reshape(batch_size, 2*self.n_ex)
        x = self.ex_embeddings(x_id)  # (batch_size, d_e)
        
        # Temporal propagation
        kci = torch.mm(ex, self.ex_graph).reshape(-1)  # (batch_size * n_kc)
        x = x.unsqueeze(1).repeat(1, self.n_kc, 1).reshape(batch_size*self.n_kc, self.d_e)
        h = h.reshape(batch_size*self.n_kc, self.d_h)        
        new_h = (1 - kci.unsqueeze(-1).repeat(1, self.d_h)) * h + kci.unsqueeze(-1).repeat(1, self.d_h) * self.tgru(x, h)
        new_h = new_h.reshape(batch_size, self.n_kc, self.d_h)
        
        # Spatial propagation
        delta_h = new_h - h.reshape(batch_size, self.n_kc, self.d_h)  # (batch_size, n_kc, d_h)
        delta_h = delta_h.unsqueeze(2).repeat(1, 1, self.n_kc, 1)  # (batch_size, n_kc(i), n_kc(j), d_h)
        e = self.kc_embeddings(torch.eye(self.n_kc)) # (n_kc, d_c)
        e = e.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_kc, 1, 1)  # (batch_size, n_kc(i), n_kc(j), d_c)
        pij = torch.cat([delta_h, e], axis=-1)  # (batch_size, n_kc(i), n_kc(j), d_h + d_c)
        pij = pij.reshape(batch_size * self.n_kc * self.n_kc, -1)
        partij = torch.relu(self.fpart(pij).reshape(batch_size, self.n_kc, self.n_kc, self.d_h))
        
        # Remove partij not corresponding to current kc i
        partij = partij * kc_graph.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, self.d_h)
        partij = partij * kci.reshape(batch_size, self.n_kc, 1, 1).repeat(1, 1, self.n_kc, self.d_h)
        partj = torch.sum(partij, axis=1)  # (batch_size, n_kc(j), d_h)
              
        # Final update
        outj = torch.relu(self.ulinear(partj.reshape(batch_size * self.n_kc, self.d_h)))
        kcj = torch.mm(kci.reshape(batch_size, self.n_kc), kc_graph).reshape(batch_size*self.n_kc)
        h = new_h.reshape(batch_size*self.n_kc, self.d_h)
        h = (1 - kcj.unsqueeze(-1).repeat(1, self.d_h)) * h + kcj.unsqueeze(-1).repeat(1, self.d_h) * self.ugru(outj, h)
        h = h.reshape(batch_size, self.n_kc, self.d_h)
        return h

class SKT(torch.nn.Module):
    
    def __init__(self, n_ex, n_kc, ex_graph, d_e=8, d_c=8, d_h=8):
        super(SKT, self).__init__()

        self.n_ex = n_ex  # Number of exercises
        self.n_kc = n_kc  # Number of KCs
        self.d_e = d_e  # Dimension of (exercise-response) embeddings
        self.d_c = d_c  # Dimension of concepts (i.e. KC) embeddings
        self.d_h = d_h  # Dimension of hidden states
        
        self.ex_graph = ex_graph
        
        # Recurrent cell
        self.skt_cell = SKTCell(n_ex, n_kc, d_e, d_c, d_h, ex_graph)

        # Output layer
        self.olinear = torch.nn.Linear(d_h, 1)
        self.ex_out = torch.nn.Parameter(torch.randn(self.n_ex))

    def init_students(self, batch_size):
        self.n_st = batch_size

    def reset_students(self):
        self.h = torch.zeros(self.n_st, self.n_kc, self.d_h)
    
    def predict(self, h, ex):
        # Predict the success proba of exercise `ex` based on the current hidden state
        # Input:
        #   h: Tensor of shape (batch_size, n_kc, d_e)
        #   ex: Tensor of shape (batch_size, n_ex)
        # Output:
        #   y_pred: Tensor of shape (batch_size) - estimated success proba on exercise `ex` 
        #   kc_pred: Tensor of shape (batch_size, n_kc) - estimated skill level on each KC
        batch_size = ex.shape[0]

        assert batch_size == self.n_st

        # To be able to predict the progress on each KC, we change
        # the prediction layer of the original SKT model
        kc_pred = self.olinear(h.reshape(batch_size*self.n_kc, self.d_h)).reshape(batch_size, self.n_kc)   

        # We predict exercise success proba using the masked softmin
        kc_mask = torch.mm(ex, self.ex_graph)  # (batch_size, n_kc)
        y_pred = torch.sigmoid(
            masked_softmin(kc_pred, kc_mask) + torch.mm(
                ex, self.ex_out.unsqueeze(-1)
            ).reshape(batch_size)
        )
        return y_pred, kc_pred

    @classmethod
    def prepare_data(cls, file, split_id=300, limit_n_st=300):
        
        _, data_ex_graph, data_exs, data_suc, _ = pk.load(
            open(file, 'rb')
        )

        T, n_st = data_suc.shape
        n_ex, n_kc = data_ex_graph.shape

        ex_features = torch.zeros(n_st, T, n_ex)

        data_y = torch.zeros(n_st, T)

        for st in range(n_st):
            for t in range(T):
                ex = int(data_exs[t, st])

                # EX Features
                ex_features[st, t, ex] = 1.

                # Output
                data_y[st, t] = data_suc[t, st]

        data_x = ex_features

        # Random permutation of the students
        perm = np.random.permutation(n_st)
        data_x = data_x[perm]
        data_y = data_y[perm]
        
        train_x = data_x[:limit_n_st, :]
        train_y = data_y[:limit_n_st, :]

        test_x = data_x[split_id:, :]
        test_y = data_y[split_id:, :]

        return (train_x, train_y, test_x, test_y)


class SKTWrapper():
    
    def __init__(self, skt):
        # Wrapper class meant to interface DKT for planning with the MBT model
        # Input:
        #   dkt: an instance of DKT model
        self.skt = skt
        self.skt.eval()
        self.n_kc = skt.n_kc
        self.n_ex = skt.n_ex
        self.d_h = skt.d_h

        self.ex_graph = skt.ex_graph
    
    def new_student(self):
        # Init one student
        self.skt.init_students(1)
        self.skt.reset_students()

    def update(self, ex, su):
        # Update the estimation of the student knoweldge state based on the (exercise-response) pair
        # Input:
        #   ex: int - taken exercise
        #   su: float - typically 0 or 1, success on the exercise

        self.skt.h = self.skt.skt_cell(
            self.skt.h,
            torch.eye(self.n_ex)[ex:ex+1],
            torch.tensor([su])
        )
    
    def estimate_level(self, h=None):
        # Estimate the current proficiency on each KC
        # Output:
        #   Tensor of shape (n_kc)
        if h is None:
            h = self.skt.h
        return torch.sigmoid(self.skt.olinear(h.reshape(self.skt.n_st, self.n_kc, self.d_h)).reshape(self.skt.n_st, self.n_kc))

    def estimate_future_level(self, ex):
        # Estimate the expected future proficiency on each KC
        # Output:
        #   Tensor of shape (n_kc)

        # Estimate the probability of success
        ex_v = torch.eye(self.n_ex)[ex:ex+1]
        success_proba, _ = self.skt.predict(self.skt.h, ex_v)
    
        # Estimate the future hidden state in case of success
        h_su = self.skt.skt_cell(
            self.skt.h,
            torch.eye(self.n_ex)[ex:ex+1],
            torch.tensor([1.])
        )

        # Estimate the future hidden state in case of failure
        h_fa = self.skt.skt_cell(
            self.skt.h,
            torch.eye(self.n_ex)[ex:ex+1],
            torch.tensor([0.])
        )   
        
        # Compute the expectation of future profiency
        return (
            success_proba * self.estimate_level(h_su) \
            + (1 - success_proba) * self.estimate_level(h_fa)
        )