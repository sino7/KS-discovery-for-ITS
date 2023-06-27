import torch
import numpy as np

class MBT():
    
    def __init__(self, student_model):
        
        self.n_kc = student_model.n_kc
        self.n_ex = student_model.n_ex
        self.ex_graph = student_model.ex_graph
        self.student_model = student_model
    
    def new_student(self):
        
        with torch.no_grad():
            self.student_model.new_student()
        
    def update_model(self, ex, su):
        
        with torch.no_grad():
            
            # If needed, update the model
            self.student_model.update(ex, su)
        
    def simulate_exercise(self, ex):
        
        with torch.no_grad():

            # Current level estimated by the model
            base_level = self.student_model.estimate_level()

            # Future level estimated by the model
            new_level = self.student_model.estimate_future_level(ex)
        
        return torch.mean(new_level - base_level)
        
    def choose_action(self):
        
        rewards = torch.zeros(self.n_ex)
        
        # Simulate different actions
        for ex in range(self.n_ex):
            rewards[ex] = self.simulate_exercise(ex)

        probas = torch.nn.Softmax(dim=0)(rewards*100).numpy()
        ex = np.random.choice(np.arange(self.n_ex), p=probas)
        
        return ex