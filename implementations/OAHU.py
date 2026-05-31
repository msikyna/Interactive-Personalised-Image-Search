import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OAHU(nn.Module):
    def __init__(self, input_dim, hidden_dims, embed_dim, 
                 V=0.99, B=0.1, g=0.1, learning_rate=1e-3, initial_matrix=None):
        """
        Parameters:
          input_dim: Dimension of input features.
          hidden_dims: List of hidden layer sizes (defines the shared backbone).
                      There will be len(hidden_dims)+1 branches:
                      branch 0 uses the raw input; branch i (i>=1)
                      uses the activation from hidden layer i.
          embed_dim: Dimension of the learned metric embedding.
                     If an initial_matrix is provided, embed_dim must equal input_dim.
          V: Discount factor for Hedge update (in (0,1)).
          B: Smooth factor; minimal allowed weight.
          g: Control parameter (should be in (0, 2/3)).
          learning_rate: Learning rate for gradient descent.
          initial_matrix: (Optional) A d×d positive‐definite matrix (as a numpy array or torch tensor)
                          to initialize the Mahalanobis metric in branch 0. If provided, embed_dim must equal input_dim.
        """
        super(OAHU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embed_dim = embed_dim
        self.V = V
        self.B = B
        self.g = g
        # Total number of branches = 1 (input branch) + number of hidden layers.
        self.num_branches = 1 + len(hidden_dims)
        
        # Build shared hidden layers (the backbone)
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
        
        # Build independent embedding layers for each branch.
        # Branch 0: input → embedding.
        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, embed_dim, bias=False))
        # For branch i (i>=1): use the activation from hidden layer i.
        for h in hidden_dims:
            self.embedding_layers.append(nn.Linear(h, embed_dim, bias=False))
        
        # If an initial matrix is provided, initialize branch 0's weight accordingly.
        if initial_matrix is not None:
            if embed_dim != input_dim:
                raise ValueError("When providing an initial_matrix, embed_dim must equal input_dim.")
            if not torch.is_tensor(initial_matrix):
                initial_matrix = torch.tensor(initial_matrix, dtype=torch.float32)
            # Compute the Cholesky factor L0 such that M0 = L0ᵀL0.
            try:
                L0 = torch.linalg.cholesky(initial_matrix)
            except Exception as e:
                raise ValueError("Provided initial_matrix is not positive definite.") from e
            # Set branch 0's weight to L0.
            self.embedding_layers[0].weight.data = L0.clone()
        
        # Initialize branch weights U as a 1D tensor (one per branch).
        # Initially, they are uniformly distributed.
        self.U = torch.ones(self.num_branches) / self.num_branches
        
        # Use one optimizer for all parameters.
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def _to_tensor(self, x):
        """Convert numpy array to torch tensor if necessary."""
        if not torch.is_tensor(x):
            return torch.tensor(x, dtype=torch.float32)
        return x

    def forward(self, x):
        """
        Given a batch x (batch_size x input_dim), compute the embedding
        from each branch. Branch 0 uses x directly; branch i uses the
        activation after the i-th hidden layer.
        Returns a list of embeddings (each normalized to unit norm),
        one per branch.
        """
        embeddings = []
        # Branch 0: raw input.
        emb0 = self.embedding_layers[0](x)
        emb0 = F.normalize(emb0, p=2, dim=1)
        embeddings.append(emb0)
        
        # Compute hidden activations sequentially.
        h = x
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(layer(h))
            emb = self.embedding_layers[i+1](h)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb)
        return embeddings
    
    def compute_loss(self, f_q, f_pos, f_neg):
        """
        Compute the Adaptive-Bound Triplet Loss (ABTL) for one branch.
        Let d_pos = ||f_q - f_pos|| and d_neg = ||f_q - f_neg|| (both in [0,2]
        since f's are unit normalized). Then we set (using a simple linear
        interpolation that meets the boundary conditions in the paper):
        
            tau_sim    = (g/2) * d_pos      (so that when d_pos=0, tau_sim=0;
                                             when d_pos=2, tau_sim=g)
            tau_dissim = 2 - (g/2)* d_neg    (so that when d_neg=0, tau_dissim=2;
                                             when d_neg=2, tau_dissim=2-g)
        
        Then the attractive loss is: L_attract = max(0, d_pos - tau_sim)
             and the repulsive loss is: L_repulse  = max(0, tau_dissim - d_neg).
        The branch loss is the average: L = 0.5 * (L_attract + L_repulse)
        (averaged over the batch).
        """
        d_pos = torch.norm(f_q - f_pos, p=2, dim=1)
        d_neg = torch.norm(f_q - f_neg, p=2, dim=1)
        tau_sim = (self.g / 2.0) * d_pos
        tau_dissim = 2 - (self.g / 2.0) * d_neg
        L_attract = F.relu(d_pos - tau_sim)
        L_repulse = F.relu(tau_dissim - d_neg)
        loss = 0.5 * (L_attract + L_repulse)
        return loss.mean()
    
    def fit(self, pairs, labels, epochs=1):
        """
        Train the OAHU model using online (or mini-batch) updates.
        The training input is assumed to be given as in your ITML code:
          - pairs is a list of tuples (query, file_feature), where for every query
            two pairs are provided: the first with label +1 (similar) and the second
            with label -1 (dissimilar).
          - labels is a list [1, -1, 1, -1, ...].
        Thus, every two consecutive pairs form one triplet: (query, positive, negative).
        """
        # Convert pairs into triplets.
        triplets = []
        for i in range(0, len(pairs), 2):
            q, pos = pairs[i]
            q2, neg = pairs[i+1]
            triplets.append((q, pos, neg))
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            # Process one triplet at a time.
            for (q, pos, neg) in triplets:
                # Convert inputs to torch tensors if necessary.
                q = self._to_tensor(q)
                pos = self._to_tensor(pos)
                neg = self._to_tensor(neg)
                # Ensure each input is 2D (batch_size=1)
                q = q.unsqueeze(0)
                pos = pos.unsqueeze(0)
                neg = neg.unsqueeze(0)
                
                self.optimizer.zero_grad()
                # Forward pass: get embeddings from all branches.
                f_q_list   = self.forward(q)
                f_pos_list = self.forward(pos)
                f_neg_list = self.forward(neg)
                
                branch_losses = []
                for l in range(self.num_branches):
                    loss_branch = self.compute_loss(f_q_list[l], f_pos_list[l], f_neg_list[l])
                    branch_losses.append(loss_branch)
                # Compute overall loss as weighted sum of branch losses.
                U_tensor = self.U.to(q.device)
                total_triplet_loss = sum(U_tensor[l] * branch_losses[l] for l in range(self.num_branches))
                total_triplet_loss.backward()
                self.optimizer.step()
                
                # Adaptive Hedge Update for branch weights.
                for l in range(self.num_branches):
                    self.U[l] = self.U[l] * (1 - (1 - self.V) * branch_losses[l].item())
                    self.U[l] = max(self.U[l], self.B / self.num_branches)
                sum_U = sum(self.U)
                self.U = torch.tensor([u / sum_U for u in self.U], dtype=torch.float32)
                
                total_loss += total_triplet_loss.item()
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(triplets):.4f}")
        return self
    
    def partial_fit(self, q, pos, neg):
        """
        Perform a single online update using one triplet (q, pos, neg).
        Each input is a 1D numpy array or torch tensor.
        """
        self.optimizer.zero_grad()
        q = self._to_tensor(q).unsqueeze(0)
        pos = self._to_tensor(pos).unsqueeze(0)
        neg = self._to_tensor(neg).unsqueeze(0)
        f_q_list   = self.forward(q)
        f_pos_list = self.forward(pos)
        f_neg_list = self.forward(neg)
        branch_losses = []
        for l in range(self.num_branches):
            loss_branch = self.compute_loss(f_q_list[l], f_pos_list[l], f_neg_list[l])
            branch_losses.append(loss_branch)
        U_tensor = self.U.to(q.device)
        total_triplet_loss = sum(U_tensor[l] * branch_losses[l] for l in range(self.num_branches))
        total_triplet_loss.backward()
        self.optimizer.step()
        
        for l in range(self.num_branches):
            self.U[l] = self.U[l] * (1 - (1 - self.V) * branch_losses[l].item())
            self.U[l] = max(self.U[l], self.B / self.num_branches)
        sum_U = sum(self.U)
        self.U = torch.tensor([u / sum_U for u in self.U], dtype=torch.float32)
        return total_triplet_loss.item()
    
    def get_mahalanobis_matrix(self):
        """
        Return the current learned Mahalanobis matrix from branch 0.
        (Since branch 0 is linear: f(x)=normalize(L0*x), we have M = L0ᵀL0.)
        """
        L0 = self.embedding_layers[0].weight.data  # shape (embed_dim, input_dim)
        M = L0.t().mm(L0)
        return M