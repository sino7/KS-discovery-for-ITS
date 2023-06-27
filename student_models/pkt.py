import numpy as np
import torch
from utils import masked_softmin_elo, elo_proba

import pickle as pk

class PKTCell(torch.nn.Module):

    def __init__(self, n_st, n_kc, n_ex, ex_graph, log_su_speed_mean=0., log_fa_speed_mean=0.):
        super(PKTCell, self).__init__()

        self.n_st = n_st
        self.n_kc = n_kc
        self.n_ex = n_ex

        self.log_su_speed = torch.nn.Parameter(torch.zeros(n_st)+log_su_speed_mean)
        self.log_fa_speed = torch.nn.Parameter(torch.zeros(n_st)+log_fa_speed_mean)
        self.initial_kc_elo = torch.nn.Parameter(1500. + torch.randn(n_st, n_kc))
        self.base_elo = torch.nn.Parameter(1500. + torch.randn(n_st))

        self.ex_graph = ex_graph

    def forward(self, h, x, su):
        # Update the recurrent state h based on x
        # The recurrent state directly contains the estimated level on each KC
        # The forward pass simulates the progress suscitated by practice
        # Input:
        #   h: (su_features, fa_features)
        #       su_features: Tensor of shape (batch_size, n_kc) - number of success on each feature
        #       fa_features: Tensor of shape (batch_size, n_kc) - number of failure on each feature
        #   x: Tensor of shape (batch_size, n_ex) - the exercise taken
        #   su: Tensor of shape (batch_size) - whether the exercise is successfully completed
        su_features, fa_features = h
        batch_size, _ = su_features.shape

        # Simulate the progress induced by practice
        kcs = torch.mm(x, self.ex_graph)
        su_features = su_features + kcs * su.unsqueeze(-1).repeat(1, self.n_kc)
        fa_features = fa_features + kcs * (1-su).unsqueeze(-1).repeat(1, self.n_kc)

        return (su_features, fa_features)


class PKT(torch.nn.Module):
    
    def __init__(self, n_kc, n_ex, ex_graph, kc_gamma, n_st=1, lr=0.1):
        super(PKT, self).__init__()

        self.n_st = n_st  # Number of students
        self.n_kc = n_kc  # Number of KCs
        self.n_ex = n_ex  # Number of Exs
        self.ex_graph = ex_graph

        self.kc_gamma = torch.nn.Parameter(kc_gamma) # Estimated structure

        # General parameters
        self.ex_elo = torch.nn.Parameter(1500. + torch.randn(n_ex))
        self.log_su_speed_mean = torch.nn.Parameter(torch.randn(1))
        self.log_fa_speed_mean = torch.nn.Parameter(torch.randn(1))
        self.a = torch.nn.Parameter(torch.Tensor([-1.]))
        self.b = torch.nn.Parameter(torch.Tensor([1.]))
        
        # Student-relative parameters
        self.pkt_cell = PKTCell(n_st, n_kc, n_ex, ex_graph)

        self.reset_optimizer(lr)
        
    def reset_optimizer(self, lr=0.1):
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.reset_student_optimizer()

    def reset_student_optimizer(self):
        self.val_opt = torch.optim.Adam(self.pkt_cell.parameters(), lr=self.opt.param_groups[-1]['lr'])

    def init_students(self, n_st):
        self.n_st = n_st
        self.pkt_cell = PKTCell(n_st, self.n_kc, self.n_ex, self.ex_graph)

    def reset_students(self):
        # Reset the recurrent states
        self.su_features = torch.zeros(self.n_st, self.n_kc)
        self.fa_features = torch.zeros(self.n_st, self.n_kc)

    def predict_kcs(self, su_features, fa_features):
        # Predict for each student the skill level on each kc
        # Input:
        #   su_features: Tensor of shape (n_st, n_kc) - number of times each KC has been succeeded
        #   fa_features: Tensor of shape (n_st, n_kc) - number of times each KC has been failed
        # Output:
        #   kc_pred: Tensor of shape (batch_size, n_kc) - estimated skill level on each KC
        kc_elos = self.pkt_cell.initial_kc_elo
        su_speed = torch.exp(self.pkt_cell.log_su_speed).unsqueeze(-1)
        fa_speed = torch.exp(self.pkt_cell.log_fa_speed).unsqueeze(-1)
        kc_elos = kc_elos + su_speed.repeat(1, self.n_kc) * su_features
        kc_elos = kc_elos + fa_speed.repeat(1, self.n_kc) * fa_features
        return kc_elos

    def predict(self, st_features, ex_features, su_features, fa_features):
        # Predict the success proba of exercise `ex` based on the current hidden state
        # Input:
        #   st_features: Tensor of shape (batch_size, n_st) - student indices
        #   ex_features: Tensor of shape (batch_size, n_ex) - exercises indices
        #   su_features: Tensor of shape (batch_size, n_kc) - number of times each KC has been succeeded
        #   fa_features: Tensor of shape (batch_size, n_kc) - number of times each KC has been failed
        # Output:
        #   y_pred: Tensor of shape (batch_size) - estimated success proba on exercise `ex_features` 
        #   kc_pred: Tensor of shape (batch_size, n_kc) - estimated skill level on each KC
        batch_size = st_features.shape[0]

        # Student current levels
        kc_elos = torch.mm(st_features, self.pkt_cell.initial_kc_elo)
        su_speed = torch.mm(st_features, torch.exp(self.pkt_cell.log_su_speed).unsqueeze(-1))
        fa_speed = torch.mm(st_features, torch.exp(self.pkt_cell.log_fa_speed).unsqueeze(-1))
        kc_elos = kc_elos + su_speed.repeat(1, self.n_kc) * su_features
        kc_elos = kc_elos + fa_speed.repeat(1, self.n_kc) * fa_features

        # Bound kc elos to 500-2500
        kc_elos = torch.min(
            torch.max(
                kc_elos,
                500. +  torch.zeros_like(kc_elos)
            ),
            2500. + torch.zeros_like(kc_elos)
        )

        # Set of parent KCs
        kcs = torch.mm(ex_features, self.ex_graph)
        kc_graph = torch.sigmoid(self.kc_gamma) * (1-torch.eye(self.n_kc))
        pcks = torch.mm(kcs, kc_graph + torch.eye(self.n_kc))  # shape (batch_size, n_kc)

        # Exercise difficulty
        ex_elo = torch.mm(ex_features, self.ex_elo.unsqueeze(-1)).squeeze(-1)  # shape (batch_size)

        # Estimate success rate
        elo = masked_softmin_elo(kc_elos, pcks)
        y_pred = elo_proba(elo, ex_elo)  # shape (batch_size)

        # Estimate KC elos
        kc_pred = torch.sigmoid(
            (
                kc_elos- 1500
            ) * np.log(10) / 400
        )

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
        
        st_features = torch.zeros(n_st, T, n_st)
        ex_features = torch.zeros(n_st, T, n_ex)
        su_features = torch.zeros(n_st, T+1, n_kc)
        fa_features = torch.zeros(n_st, T+1, n_kc)

        y = torch.zeros(n_st, T)

        for st in range(n_st):
            for t in range(T):
                ex = int(data_exs[t, st])
                su = data_suc[t, st]

                # ST Features
                st_features[st, t, st] = 1.

                # EX Features
                ex_features[st, t, ex] = 1.

                # SU Features
                su_features[st, t+1] = su_features[st, t]
                su_features[st, t+1] += su * data_ex_graph[ex]

                # FA Features
                fa_features[st, t+1] = fa_features[st, t]
                fa_features[st, t+1] += (1-su) * data_ex_graph[ex]

                # Output
                y[st, t] = su
                
        # Random permutation before train/test split
        perm = np.random.permutation(n_st)
        st_features = st_features[perm]
        ex_features = ex_features[perm]
        su_features = su_features[perm]
        fa_features = fa_features[perm]
        y = y[perm]
                
        st_train = st_features[:limit_n_st]
        ex_train = ex_features[:limit_n_st]
        su_train = su_features[:limit_n_st]
        fa_train = fa_features[:limit_n_st]
        y_train = y[:limit_n_st]

        st_test = st_features[split_id:]
        ex_test = ex_features[split_id:]
        su_test = su_features[split_id:]
        fa_test = fa_features[split_id:]
        y_test = y[split_id:]
        
        st_train = st_train.reshape(limit_n_st * T, n_st)
        ex_train = ex_train.reshape(limit_n_st * T, n_ex)
        su_train = su_train[:, :-1].reshape(limit_n_st * T, n_kc)
        fa_train = fa_train[:, :-1].reshape(limit_n_st * T, n_kc)
        y_train = y_train.reshape(limit_n_st * T)

        st_test = st_test.reshape((n_st - split_id) * T, n_st)
        ex_test = ex_test.reshape((n_st - split_id) * T, n_ex)
        su_test = su_test[:, :-1].reshape((n_st - split_id) * T, n_kc)
        fa_test = fa_test[:, :-1].reshape((n_st - split_id) * T, n_kc)
        y_test = y_test.reshape((n_st - split_id) * T)

        # Permutation fo the training examples
        tr_perm = np.random.permutation(limit_n_st * T)
        st_train_perm = st_train[tr_perm]
        ex_train_perm = ex_train[tr_perm]
        su_train_perm = su_train[tr_perm]
        fa_train_perm = fa_train[tr_perm]
        y_train_perm = y_train[tr_perm]
        
        return (
            st_train_perm, ex_train_perm, su_train_perm, fa_train_perm, y_train_perm,
            st_test, ex_test, su_test, fa_test, y_test
        )
    
class PKTWrapper():

    def __init__(self, pkt):

        self.pkt = pkt
        self.pkt.eval()
        self.n_kc = pkt.n_kc
        self.n_ex = pkt.n_ex

        self.ex_graph = pkt.ex_graph

    def new_student(self):
        # Init one student
        self.pkt.init_students(1)
        self.pkt.reset_students()

    def update(self, ex, su):
        # Update the estimation of the student knoweldge state based on the (exercise-response) pair
        # Input:
        #   ex: int - taken exercise
        #   su: float - typically 0 or 1, success on the exercise

        ex_v = torch.eye(self.n_ex)[ex:ex+1]
        su_v = torch.tensor([su])

        self.pkt.su_features, self.pkt.fa_features = self.pkt.pkt_cell(
            (self.pkt.su_features, self.pkt.fa_features),
            ex_v,
            su_v
        )
    
    def estimate_level(self, h=None):
        # Estimate the current proficiency on each KC
        # Output:
        #   Tensor of shape (n_kc)
        
        if h is None:
            h = (self.pkt.su_features, self.pkt.fa_features)
            
        su_features, fa_features = h

        # Estimate KC elos
        kc_pred = torch.sigmoid(
            (
                self.pkt.predict_kcs(su_features, fa_features)- 1500
            ) * np.log(10) / 400
        )
        return kc_pred

    def estimate_future_level(self, ex):
        # Estimate the expected future proficiency on each KC
        # Output:
        #   Tensor of shape (n_kc)

        # Estimate the probability of success
        st_v = torch.eye(1)
        ex_v = torch.eye(self.n_ex)[ex:ex+1]
        success_proba, _ = self.pkt.predict(st_v, ex_v, self.pkt.su_features, self.pkt.fa_features)
    
        # Estimate the skill levels in case of success
        su_su_features, su_fa_features = self.pkt.pkt_cell((self.pkt.su_features, self.pkt.fa_features), ex_v, torch.Tensor([1.]))
        su_level = self.estimate_level((su_su_features, su_fa_features))

        # Estimate the future skill levels in case of failure
        fa_su_features, fa_fa_features = self.pkt.pkt_cell((self.pkt.su_features, self.pkt.fa_features), ex_v, torch.Tensor([0.]))
        fa_level = self.estimate_level((fa_su_features, fa_fa_features))   
        
        # Compute the expectation of future profiency
        return (
            success_proba * su_level \
            + (1 - success_proba) * fa_level
        )