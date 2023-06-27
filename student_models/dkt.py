import numpy as np
import torch
from utils import masked_softmin

import pickle as pk

class DKTCell(torch.nn.Module):

    def __init__(self, n_kc, n_ex, d_h=16):

        super(DKTCell, self).__init__()

        self.n_kc = n_kc
        self.n_ex = n_ex
        self.d_h = d_h
        
        self.lstm = torch.nn.LSTMCell(n_ex*2+n_kc, d_h)

    def forward(self, hc, x):
        # Update the recurrent state (h, c) based on x
        hc = self.lstm(x, hc)
        return hc

class DKT(torch.nn.Module):
    
    def __init__(self, n_kc, n_ex, ex_graph, d_h=16, lr=0.1):
        super(DKT, self).__init__()
        
        self.n_kc = n_kc  # Number of KCs
        self.n_ex = n_ex  # Number of Exs
        self.d_h = d_h
        
        self.ex_graph = ex_graph
        
        self.dkt_cell = DKTCell(n_kc, n_ex, d_h)
        self.out = torch.nn.Linear(d_h, n_kc)
        self.ex_b = torch.nn.Parameter(torch.randn(n_ex))

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def init_students(self, batch_size):
        self.n_st = batch_size

    def reset_students(self):
        # Reset the recurrent states
        self.h = torch.zeros(self.n_st, self.d_h)
        self.c = torch.zeros(self.n_st, self.d_h)

    def predict(self, hc, ex):
        # Predict the success proba of exercise `ex` based on the current hidden state
        # Input:
        #   hc: (h, c)
        #       h: Tensor of shape (batch_size, d_h)
        #       c: Tensor of shape (batch_size, d_h)
        #   ex: Tensor of shape (batch_size, n_ex)
        # 
        # Output:
        #   y_pred: Tensor of shape (batch_size) - estimated success proba on exercise `ex` 
        #   kc_pred: Tensor of shape (batch_size, n_kc) - estimated skill level on each KC
        batch_size = ex.shape[0]
        h, _ = hc
        assert batch_size == self.n_st

        # To be able to predict the progress on each KC, we change
        # the prediction layer of the original DKT model
        kc_pred = self.out(h)

        # We predict exercise success proba using the masked softmin
        kc_mask = torch.mm(ex, self.ex_graph)  # (batch_size, n_kc)
        y_pred = torch.sigmoid(
            masked_softmin(kc_pred, kc_mask) + torch.mm(
                ex, self.ex_b.unsqueeze(-1)
            ).reshape(batch_size)
        )
        return y_pred, kc_pred

    def forward(self, x, ex):
        # Forward pass through the DKT model
        # Input: 
        #   x: Tensor of shape (batch_size, T, 2*n_ex + n_kc) - sequences of (exercise-response) pairs
        #   ex: Tensor of shape (batch_size, T, n_ex) - sequences of taken exercises
        # Output:
        #   y_pred: Tensor of shape (batch_size, T) - estimated success proba on exercises
        #   kc_pred: Tensor of shape (batch_size, T, n_kc) - estimated skill levels on each KC
        batch_size, T, _ = x.shape
        assert batch_size == self.n_st

        y_pred = torch.zeros(batch_size, T)
        kc_pred = torch.zeros(batch_size, T, self.n_kc)
        
        for t in range(T):
            
            # Prediction at time step t
            y_pred_t, kc_pred_t = self.predict((self.h, self.c), ex[:, t])
            y_pred[:,t] = y_pred_t
            kc_pred[:,t] = kc_pred_t

            # Update the recurrent states based on the student success
            self.h, self.c = self.dkt_cell((self.h, self.c), x[:, t])
            
        return y_pred, kc_pred
    
    @classmethod
    def prepare_data(cls, file, split_id=300, limit_n_st=300):
        # Class method used to prepare the data for training a DKT model
        # Input:
        #   file: path to the file storing the data
        #   split_id: int - training/test split
        #   limit_n_st: int - maximum number of student trajectories to use for training
        # Output:
        #   train_x: Tensor of shape (limit_n_st, T, n_ex*2+n_kc)
        #   train_ex: Tensor of shape (limit_n_st, T, n_ex)
        #   train_y: Tensor of shape (limit_n_st, T)
        #   test_x: Tensor of shape (N-split_id, T, n_ex*2+n_kc)
        #   test_ex: Tensor of shape (N-split_id, T, n_ex)
        #   test_y: Tensor of shape (N-split_id, T)

        _, data_ex_graph, data_exs, data_suc, _ = pk.load(
            open(file, 'rb')
        )

        T, n_st = data_suc.shape
        n_ex, n_kc = data_ex_graph.shape
        
        ex_features = torch.zeros(n_st, T, n_ex)
        kc_features = torch.zeros(n_st, T, n_kc)

        data_y = torch.zeros(n_st, T)

        for st in range(n_st):
            for t in range(T):
                ex = int(data_exs[t, st])

                # EX Features
                ex_features[st, t, ex] = 1.

                # KC Features
                kc_features[st, t] = data_ex_graph[ex]

                # Output
                data_y[st, t] = data_suc[t, st]
                
        data_x = torch.cat([ex_features, kc_features, ex_features * data_y.unsqueeze(-1).repeat(1, 1, n_ex)], axis=2)
        data_ex = ex_features

        # Random permutation of the students
        perm = np.random.permutation(n_st)
        data_x = data_x[perm]
        data_ex = data_ex[perm]
        data_y = data_y[perm]
        
        # Train / test split 
        train_x = data_x[:limit_n_st]
        train_ex = data_ex[:limit_n_st]
        train_y = data_y[:limit_n_st]

        test_x = data_x[split_id:]
        test_ex = data_ex[split_id:]
        test_y = data_y[split_id:]
        
        return (train_x, train_ex, train_y, test_x, test_ex, test_y)
    
class DKTWrapper():

    def __init__(self, dkt):
        # Wrapper class meant to interface DKT for planning with the MBT model
        # Input:
        #   dkt: an instance of DKT model
        self.dkt = dkt
        self.dkt.eval()
        self.n_kc = dkt.n_kc
        self.n_ex = dkt.n_ex
        self.d_h = dkt.d_h

        self.ex_graph = dkt.ex_graph

    def new_student(self):
        # Init one student
        self.dkt.init_students(1)
        self.dkt.reset_students()

    def update(self, ex, su):
        # Update the estimation of the student knoweldge state based on the (exercise-response) pair
        # Input:
        #   ex: int - taken exercise
        #   su: float - typically 0 or 1, success on the exercise

        x = torch.zeros(1, self.n_ex*2+self.n_kc)
        x[0, ex] = 1.
        x[0, self.n_ex+self.n_kc+ex] = su
        x[0, self.n_ex:self.n_ex+self.n_kc] = self.ex_graph[ex]
        
        self.dkt.h, self.dkt.c = self.dkt.dkt_cell((self.dkt.h, self.dkt.c), x)
    
    def estimate_level(self, h=None):
        # Estimate the current proficiency on each KC
        # Output:
        #   Tensor of shape (n_kc)
        if h is None:
            h = self.dkt.h
        return torch.sigmoid(self.dkt.out(h))

    def estimate_future_level(self, ex):
        # Estimate the expected future proficiency on each KC
        # Output:
        #   Tensor of shape (n_kc)

        # Estimate the probability of success
        ex_v = torch.eye(self.n_ex)[ex:ex+1]
        success_proba, _ = self.dkt.predict((self.dkt.h, self.dkt.c), ex_v)
    
        # Estimate the future hidden state in case of success
        x_su = torch.zeros(1, self.n_ex*2+self.n_kc)
        x_su[0, ex] = 1.
        x_su[0, self.n_ex+self.n_kc+ex] = 1.
        x_su[0, self.n_ex:self.n_ex+self.n_kc] = self.ex_graph[ex]
        h_su, _ = self.dkt.dkt_cell((self.dkt.h, self.dkt.c), x_su)

        # Estimate the future hidden state in case of failure
        x_fa = torch.zeros(1, self.n_ex*2+self.n_kc)
        x_fa[0, ex] = 1.
        x_fa[0, self.n_ex+self.n_kc+ex] = 0.
        x_fa[0, self.n_ex:self.n_ex+self.n_kc] = self.ex_graph[ex]
        h_fa, _ = self.dkt.dkt_cell((self.dkt.h, self.dkt.c), x_fa)        
        
        # Compute the expectation of future profiency
        return (
            success_proba * self.estimate_level(h_su) \
            + (1 - success_proba) * self.estimate_level(h_fa)
        )