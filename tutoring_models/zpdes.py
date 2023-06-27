import torch
import numpy as np


class ZPDES():
    
    def __init__(self, kc_graph, ex_graph):
        
        self.n_ex = ex_graph.shape[0]
        self.n_kc = kc_graph.shape[0]
        self.kc_graph = kc_graph
        self.ex_graph = ex_graph
        
        self.activated_kcs = torch.zeros(self.n_kc)
        self.validated_kcs = torch.zeros(self.n_kc)

        self.activated_exs = torch.zeros(self.n_ex)
        
        self.sliding_ex_success = torch.zeros(self.n_ex)
        self.reward = torch.zeros(self.n_ex)
        
        self.deactivation_threshold = 0.9
        self.validation_threshold = 0.6
        
        self.alpha = 0.5
        self.beta = 0.5
        
        self.update_exercise_repertoire()
        
    def update_exercise_repertoire(self):
        
        # Check which exercise reached validation threshold
        validated_exs = (self.sliding_ex_success > self.validation_threshold).float()
        
        # Check which KCs are thus validated
        self.validated_kcs += torch.mm(validated_exs.unsqueeze(0), self.ex_graph).squeeze(0)
        self.validated_kcs = (self.validated_kcs > 0).float()
                
        # Check which KCs are unlocked: a KC is unlocked if all its parents are validated
        unlocked_parents_count = torch.mm(self.validated_kcs.unsqueeze(0), self.kc_graph).squeeze(0)
        parents_count = torch.mm(torch.ones(1, self.n_kc), self.kc_graph).squeeze(0)
        parents_count = torch.sum(self.kc_graph, axis=0)
        self.activated_kcs += (unlocked_parents_count == parents_count).float()
        self.activated_kcs = (self.activated_kcs > 0).float()
        
        # Check wich exercises are unlocked: an exercise is unlocked if all its corresponding KCs are activated
        unlocked_parents_count = torch.mm(self.activated_kcs.unsqueeze(0), self.ex_graph.T).squeeze(0)
        parents_count = torch.mm(torch.ones(1, self.n_kc), self.ex_graph.T).squeeze(0)
        self.activated_exs += (unlocked_parents_count == parents_count).float()
        self.activated_exs = (self.activated_exs > 0).float()
        
        # Check if some exercises should be deactivated
        easy_exercises = (self.sliding_ex_success > self.deactivation_threshold).float()
        self.activated_exs *= (1 - easy_exercises)
        
    def reset(self):
        self.activated_kcs = torch.zeros(self.n_kc)
        self.validated_kcs = torch.zeros(self.n_kc)
        self.activated_exs = torch.zeros(self.n_ex)
        self.update_exercise_repertoire()
        
    def learn_from_exercise(self, ex, success):
        
        self.reward[ex] = (1 - self.beta) * self.reward[ex] + self.beta * (success - self.sliding_ex_success[ex])
        self.sliding_ex_success[ex] = (1 - self.alpha) * self.sliding_ex_success[ex] + self.alpha * success
        self.update_exercise_repertoire()
        
    def choose_action(self):
        
        rewards = self.activated_exs + self.reward
        probas = torch.nn.Softmax(dim=0)(100*rewards).numpy()
        return np.random.choice(np.arange(self.n_ex), p=probas)