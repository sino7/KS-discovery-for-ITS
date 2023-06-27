import torch
import networkx as nx
import numpy as np


class GraphSampler():

    def __init__(self, method, n, p, q=None, verbose=False):
        
        self.method = method
        self.n = n
        self.p = p
        self.q = q
        self.verbose = verbose

    def sample(self, seed=None):

        if self.method == "ER":
            if self.verbose:
                print("Erdos-Renyi random graph sampling with N=%d and k=%d" % (self.n, self.p))
                print("The parameter k is the expected degree for each node.")
            graph = nx.generators.erdos_renyi_graph(n=self.n, p=self.p/(self.n-1), seed=seed)
            return self.adjacency_matrix(graph)

        elif self.method == "BA":
            if self.verbose:
                print("Barabasi-Albert scale-free graph sampling with N=%d and m=%d" % (self.n, self.p))
                print("The parameter m is the number of edges to attach from a new node to existing nodes.")
            graph = nx.generators.barabasi_albert_graph(n=self.n, m=self.p, seed=seed)
            return self.adjacency_matrix(graph)

        elif self.method == "WS":
            assert self.q is not None, "With Watts-Strogatz algorithm, a second parameter should be provided."
            if self.verbose:
                print("Watts-Strogatz small-world graph sampling with N=%d, k=%d and p=%d" % (self.n, self.p, self.q))
                print("The parameter k is")
                print("The parameter p is")
            graph = nx.generators.watts_strogatz_graph(n=self.n, k=self.p, p=self.q)
            return self.adjacency_matrix(graph)

    def adjacency_matrix(self, graph):
        # Creates a DAG from a undirected graph
        adjacency_matrix = torch.zeros(self.n, self.n)
        permutation = np.random.permutation(self.n)
        for (i, j) in graph.edges():
            adjacency_matrix[permutation[i], permutation[j]] = 1.

        return adjacency_matrix

def ancestor_matrix(adjacency):
    # Returns a ancestor matrix with a(i, j) being the minimum distance in the graph between i and j
    
    d = adjacency.shape[0]
    a = torch.eye(d)
    b = torch.zeros(d, d)

    for i in range(d):
        a = torch.mm(a, adjacency)
        c = ((i+1) * (a>0)).float()
        b = torch.max(b, c)
        
    return b

def has_cycle(graph):
    A = (graph>0).float()
    return torch.trace(torch.matrix_exp(A)) - A.shape[0]

def clean_graph(graph):
    A = graph.clone()
    # Removes cycles from a graph, by removing the edges of cycles with lowest predicted proba
    n_kc = A.shape[0]
    sorted_indices = sorted(np.arange(n_kc**2), key=lambda x: A[x//n_kc, x%n_kc])
    x = 0
    while has_cycle(A)>0:
        B = A.clone()
        i = sorted_indices[x] // n_kc
        j = sorted_indices[x] % n_kc
        B[i, j] = 0.
        if has_cycle(B) < has_cycle(A):
            A[i, j] = 0.
        x += 1
        
    return A

def elo_proba(elo_a, elo_b):
    # Compute the probability of a winning over b, or student a succeeding at an exercise b
    # Input:
    #   elo_a   : Float Tensor of shape (batch_size)
    #   elo_b   : Float Tensor of shape (batch_size)
    # Output:
    #   Float Tensor of shape (batch_size)
    return torch.sigmoid(
        np.log(10) * (elo_a - elo_b) / 400
    )

def masked_softmin_elo(elos, mask):
    # Computes a soft minimum over the list of provided elos
    # Input:
    #   elos    : Float Tensor of shape (batch_size, n_kc)
    #   mask    : Float Tensor of shape (batch_size, n_kc)

    assert elos.shape == mask.shape, "Input Tensors shape do not match"
    batch_size, n_kc = elos.shape
       
    weights = mask * torch.exp(-np.log(10) * elos / 400)
    normalized_weights = weights / (1e-100 + torch.sum(weights, axis=-1).unsqueeze(-1).repeat(1, n_kc))
    
    return torch.sum(
        normalized_weights * elos,
        axis=1
    )

def masked_softmin(x, mask):
    # Computes a soft minimum over the list of provided elos
    # Input:
    #   x    : Float Tensor of shape (batch_size, n_kc)
    #   mask    : Float Tensor of shape (batch_size, n_kc)

    assert x.shape == mask.shape, "Input Tensors shape do not match"
    batch_size, n_kc = x.shape
       
    weights = mask * torch.exp(-x)
    normalized_weights = weights / (1e-100 + torch.sum(weights, axis=-1).unsqueeze(-1).repeat(1, n_kc))
    
    return torch.sum(
        normalized_weights * x,
        axis=1
    )

def find_order(adjacency):
    # Function to find the ordering of the variables based on the graph adjacency matrix
    
    adjacency = (adjacency.float() > 0.5).int()
    
    a = adjacency.clone()
    num_vertices = adjacency.shape[1]
    v_list = list(range(num_vertices))
    order = []
    
    while len(v_list) > 0:
        
        # Find vertices with no parent
        no_parents = []    
        
        for v in v_list.copy():            
            if torch.sum(a[:, v])==0:
                no_parents.append(v)
        
        if len(no_parents) == 0:
            raise ValueError("This graph contains cycles")
            
        s = np.random.choice(no_parents)
        order.append(s)
        v_list.remove(s)
        a[s] = 0

    return order