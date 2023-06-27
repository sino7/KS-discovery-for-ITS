import numpy as np
import torch
import torch.nn.functional as F

import networkx as nx
from utils import *

class StudentModel():
    # Class used to simulate learner trajectories
    # It is used in our experiments both to generate synthetic data to evaluate KS discovery methods,
    # and as a simulator to evaluate the recommendations provided by tutoring models.

    def __init__(self, n_ex, n_kc, n_st):
        
        self.n_ex = n_ex  # Number of exercises
        self.n_kc = n_kc  # Number of KCs
        self.n_st = n_st  # Number of students

        # Initialize all parameters
        self.init_parameters()

        # Initialize student parameters
        self.init_student_parameters()

    def init_parameters(self):

        # KC graph
        gs = GraphSampler("ER", self.n_kc, 2)
        adjacency = gs.sample()
        ancestor = ancestor_matrix(adjacency)
        pruned_adjacency = adjacency*(ancestor<2)
        self.kc_graph = pruned_adjacency
        
        # EX graph
        self.ex_graph = torch.zeros(self.n_ex, self.n_kc)
        for k in range(self.n_ex):
            if k < self.n_kc:
                self.ex_graph[k, k] = 1.
            else:
                self.ex_graph[k, np.random.randint(self.n_kc)] = 1.
        for i in range(self.n_ex):
            is_valid = False
            while not is_valid:
                ex = np.random.randint(self.n_ex)
                kc = np.random.randint(self.n_kc)
                is_valid = torch.sum(torch.mm(self.ex_graph, ancestor+ancestor.T)[ex, kc]) == 0
            self.ex_graph[ex, kc] = 1.

        
        self.ex_elo = 1500. + 300 *  torch.randn(self.n_ex)
        self.log_success_reward = torch.Tensor([4.])
        self.log_failure_reward = torch.Tensor([2.])
        self.log_tau = torch.Tensor([1.])
        self.epsilon = 0.5

    def init_student_parameters(self):

        self.base_elo = 1000. + 200. * torch.randn(self.n_st)
        self.initial_kc_elo = self.base_elo.unsqueeze(-1).repeat(1, self.n_kc) + 200. * torch.randn(self.n_st, self.n_kc)
        self.log_lrng_speed = 3. + 0.3 * torch.randn(self.n_st)

    def reset_student_trajectories(self):

        self.st_kc_elo = self.initial_kc_elo
        self.lt_kc_elo = self.initial_kc_elo
    
    def student_learning(self, a, success):
        # Input
        #   a       : Integer Tensor of shape (batch_size), the indices of the exercises
        #   success : Boolean Tensor of shape (batch_size), whether the students completed the exercises

        # Effect of succeeding vs failing
        lrng_speed = (
            torch.exp(self.log_failure_reward) \
                + success * (
                torch.exp(self.log_success_reward) - torch.exp(self.log_failure_reward)
            )
        ) * torch.exp(self.log_lrng_speed)
        
        # Multiply by a factor representing whether the prerequisite are attained
        ex_elo = self.ex_elo[a]
        parents_graph = torch.mm(self.ex_graph[a], self.kc_graph.T)
        
        st_elo = masked_softmin_elo(self.lt_kc_elo, parents_graph)
        proba = elo_proba(st_elo, 2000.+torch.zeros_like(ex_elo))
        proba = torch.nan_to_num(proba, nan=1.)
        lrng_speed *= proba
        
        # Multiply by a factor limiting progress on exercises that are too easy or too difficult
        success_proba = self.success_prediction(a)   
        progress_area = 4 * success_proba * (1 - success_proba)
        
        # Estimate the effect of training on the KCs
        kc_delta = (lrng_speed * progress_area).unsqueeze(-1).repeat(1, self.n_kc) * self.ex_graph[a]
        
        # Learning on the long-term
        self.lt_kc_elo = self.lt_kc_elo + self.epsilon * kc_delta * torch.exp((self.lt_kc_elo - self.st_kc_elo)/100)
        self.lt_kc_elo = torch.max(self.lt_kc_elo, torch.zeros(self.n_st, self.n_kc) + 500.)
        self.lt_kc_elo = torch.min(self.lt_kc_elo, torch.zeros(self.n_st, self.n_kc) + 2500.)
        
        # Learning on the short-term
        self.st_kc_elo = self.st_kc_elo + kc_delta
        
        # Forgetting on the short-term
        self.st_kc_elo = self.lt_kc_elo + (self.st_kc_elo - self.lt_kc_elo) * torch.exp(-1/torch.exp(self.log_tau))
        self.st_kc_elo = torch.max(self.st_kc_elo, torch.zeros(self.n_st, self.n_kc) + 500.)
        self.st_kc_elo = torch.min(self.st_kc_elo, torch.zeros(self.n_st, self.n_kc) + 2500.)
        
    def success_prediction(self, a):
        # Compute the probability of the students completing the exercises
        # Input:
        #   a       : Integer Tensor of shape (batch_size), the indices of the exercises
        # Output:
        #   proba   : Float Tensor of shape (batch_size), proba of completing the exercises
                
        # Simulate the decay of the KC since the last exercise
        # TODO
        
        # Estimating current level as combination of long-term and short-term skill levels
        kc_elo_cat = torch.cat([self.st_kc_elo.unsqueeze(0), self.lt_kc_elo.unsqueeze(0)], axis=0)
        kc_elo = torch.sum(torch.nn.Softmax(dim=0)(kc_elo_cat) * kc_elo_cat, axis=0)
        
        # Estimate the probability of success
        ex_elo = self.ex_elo[a]
        parents_graph = self.ex_graph[a] + torch.mm(self.ex_graph[a], self.kc_graph.T)
        
        st_elo = masked_softmin_elo(kc_elo, parents_graph)
        proba = elo_proba(st_elo, ex_elo)
        
        return proba
    
    def simulate_exercise(self, a):
        # Simulate the students taking on one exercise
        # Input:
        #   a       : Integer Tensor of shape (batch_size), the indices of the exercises
        # Output:
        #   proba   : Float Tensor of shape (batch_size), proba of completing the exercises
        #   success : Boolean Tensor of shape (batch_size), whether the students completed the exercises
        batch_size = a.shape[0]  
        proba = self.success_prediction(a)
        success = torch.rand(batch_size) < proba
        self.student_learning(a, success)
        
        return proba, success

    def predict(self, data_exs, data_suc):

        predictions = torch.zeros_like(data_exs)
        _, T = data_exs.shape

        # Student model prediction
        for t in range(T):
            a = data_exs[:, t].int().numpy()
            
            # Predict the student successes on the exercises
            predictions[:, t] = self.success_prediction(a)
            
            # Forward pass through the model to simulate student learning 
            # from its experience with the undergone exercise
            self.student_learning(a, data_suc[:, t])
            
        return predictions

    def plot_kc_graph(self):
        # Plot kc graph
        graph = nx.from_numpy_matrix(self.kc_graph.detach().numpy(),create_using=nx.MultiDiGraph())
        nx.draw(graph, with_labels=True, node_size=500, node_color="lightgray")

    def plot_full_graph(self):
        # Plot kc graph and relations between KC and exercises
        graph = nx.from_numpy_matrix(self.kc_graph.detach().numpy(),create_using=nx.MultiDiGraph())
        for i in range(self.n_ex):
            graph.add_node("Ex %d" % i)
            for k in range(self.n_kc):
                if self.ex_graph[i, k]:
                    graph.add_edge(k, "Ex %d" % i)
        nx.draw(graph, with_labels=True, node_size=500, node_color="lightgray")