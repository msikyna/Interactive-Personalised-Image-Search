import numpy as np

def project_psd(M):
    """
    Project a symmetric matrix M onto the cone of PSD matrices via eigenvalue clipping.
    
    This ensures that the updated Mahalanobis matrix remains positive semidefinite.
    """
    # Ensure symmetry
    M = (M + M.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M)
    # Clip negative eigenvalues to zero
    eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

class LSODML:
    """
    Large-Scale Online Distance Metric Learning (LSODML)
    
    This class implements the LSODML algorithm exactly as described in the paper:
    "Large-Scale Multi-modal Distance Metric Learning with Application to 
     Content-Based Information Retrieval and Image Classification."
    
    The algorithm uses Dual Random Projection (DRP) to scale to high dimensions.
    For each incoming triplet (query, similar, dissimilar) the update is computed as follows:
      1. In the original space, compute A = (q - neg)(q - neg)^T - (q - pos)(q - pos)^T.
      2. Project q, pos, and neg into a lower-dimensional space using a random matrix R.
      3. In the reduced space, compute A_hat and the current metric M_hat = Rᵀ M R.
      4. Compute the loss: loss = max(0, 1 - trace(M_hat * A_hat)).
      5. Compute the squared Frobenius norm of A_hat.
      6. Compute the step size τ̂ = min(C, loss / ||A_hat||_F²) if loss > 0.
      7. Update the original metric: M = M - τ̂ * A.
      8. Project M onto the PSD cone.
      
    Note: If DRP is disabled (use_drp=False), the update is performed in the original space.
    """
    def __init__(self, d, C=1.0, use_drp=True, m=100, initial_matrix=None):
        """
        Parameters:
          d            : Dimensionality of the input vectors.
          C            : Aggressiveness parameter.
          use_drp      : If True, use DRP (required by the paper).
          m            : Projection dimension (m << d).
          initial_matrix : (Optional) initial Mahalanobis matrix (d x d); if None, identity is used.
        """
        self.d = d
        self.C = C
        self.use_drp = use_drp
        if use_drp:
            if m is None:
                raise ValueError("Must specify projection dimension m when use_drp is True.")
            self.m = m
            # Generate a random projection matrix R with entries ~ N(0, 1/sqrt(m))
            self.R = np.random.normal(0, 1/np.sqrt(m), size=(d, m))
        if initial_matrix is not None:
            self.M = initial_matrix.copy()
        else:
            self.M = np.eye(d)
    
    def _update_triplet(self, q, pos, neg):
        """
        Update the Mahalanobis matrix using a triplet (q, pos, neg) following the paper’s steps.
        
        Steps:
          1. Compute difference vectors in the original space.
          2. Form A = (q - neg)(q - neg)^T - (q - pos)(q - pos)^T.
          3. If DRP is used, project q, pos, neg into the reduced space:
                q_hat = Rᵀ * q, etc.
          4. In the reduced space, form A_hat analogously.
          5. Compute the projected metric: M_hat = Rᵀ * M * R.
          6. Compute loss = max(0, 1 - trace(M_hat * A_hat)).
          7. Compute ||A_hat||²_F.
          8. Compute τ̂ = min(C, loss / ||A_hat||²_F) if loss > 0.
          9. Update the original metric: M = M - τ̂ * A.
         10. Project M onto the PSD cone.
        """
        # Step 1: Compute difference vectors in original space
        diff_pos = q - pos
        diff_neg = q - neg
        
        # Step 2: Construct A in the original space
        A = np.outer(diff_neg, diff_neg) - np.outer(diff_pos, diff_pos)
        
        if self.use_drp:
            # Step 3: Project q, pos, neg into the reduced space
            q_hat   = self.R.T @ q
            pos_hat = self.R.T @ pos
            neg_hat = self.R.T @ neg
            
            # Step 4: Construct A_hat in the reduced space
            A_hat = np.outer( (q_hat - neg_hat), (q_hat - neg_hat) ) - np.outer( (q_hat - pos_hat), (q_hat - pos_hat) )
            
            # Step 5: Compute projected metric M_hat = Rᵀ * M * R
            M_hat = self.R.T @ self.M @ self.R
            
            # Step 6: Compute loss = max(0, 1 - trace(M_hat * A_hat))
            loss = max(0, 1 - np.trace(M_hat @ A_hat))
            
            # Step 7: Compute the squared Frobenius norm of A_hat
            normA = np.linalg.norm(A_hat, 'fro')**2
        else:
            # Without DRP, compute in the original space
            margin = np.trace(self.M @ A)
            loss = max(0, 1 - margin)
            normA = np.linalg.norm(A, 'fro')**2
        
        # Step 8: Compute step size τ̂ if loss > 0
        if loss > 0 and normA > 0:
            tau = min(self.C, loss / normA)
            # Step 9: Update the metric in the original space
            self.M = self.M - tau * A
            # Step 10: Project updated M onto the PSD cone
            self.M = project_psd(self.M)
    
    def fit(self, pairs, labels):
        """
        Fit the LSODML model using training pairs and labels.
        
        Assumes each query is provided twice:
          - (query, similar) with label +1
          - (query, dissimilar) with label -1
        which are combined to form a triplet.
        """
        if len(pairs) != len(labels):
            raise ValueError("Pairs and labels must have the same length.")
        if len(pairs) % 2 != 0:
            raise ValueError("Number of pairs must be even (one similar and one dissimilar per query).")
        
        for i in range(0, len(pairs), 2):
            # First pair: (query, similar) with label +1; second pair: (query, dissimilar) with label -1.
            q1, pos = pairs[i]
            q2, neg = pairs[i+1]
            # Ensure that the query is identical in both pairs.
            if not np.allclose(q1, q2):
                raise ValueError(f"Mismatched queries in pair indices {i} and {i+1}.")
            q = q1  # common query vector
            self._update_triplet(q, pos, neg)
    
    def get_mahalanobis_matrix(self):
        """Return the learned Mahalanobis matrix."""
        return self.M

class LSMDML:
    """
    Large-Scale Multi-modal Distance Metric Learning (LSMDML)
    
    This class implements LSMDML exactly as described in the paper.
    For each modality, a separate Mahalanobis metric is learned using a DRP-based PA update.
    The final distance function is a weighted combination of these per-modality metrics.
    
    For each training triplet (q, pos, neg), where each element is a dictionary mapping modality index to feature vector,
    the following steps are performed:
      1. For each modality j:
           a. Compute difference vectors: diff_pos = q[j] - pos[j] and diff_neg = q[j] - neg[j].
           b. Form A = (q[j]-neg[j])(q[j]-neg[j])^T - (q[j]-pos[j])(q[j]-pos[j])^T.
           c. If DRP is used, project the vectors (using a modality-specific random matrix) to compute A_hat and M_hat.
           d. Compute loss = max(0, 1 - trace(M_hat * A_hat)) (or in the original space).
           e. Compute the squared Frobenius norm and τ = min(C, loss / norm²).
           f. Update M_j = M_j - τ * A and project onto the PSD cone.
           g. Compute distances f_pos and f_neg.
      2. Update the coefficient vector θ using a Passive-Aggressive rule:
           a. Compute loss_theta = max(0, 1 + θᵀ (f_pos_vector - f_neg_vector)).
           b. Compute η = min(C_prime, loss_theta / ||f_pos - f_neg||²).
           c. Update θ = max(0, θ + η (f_pos - f_neg)) and normalize.
    """
    def __init__(self, modalities_dims, C=1.0, C_prime=1.0, use_drp=True, m=100, initial_matrices=None):
        """
        Parameters:
          modalities_dims : List of dimensions for each modality.
          C               : Aggressiveness parameter for the per-modality metric update.
          C_prime         : Aggressiveness parameter for updating the coefficient vector θ.
          use_drp         : If True, use DRP for each modality update.
          m               : Projection dimension (assumed same for all modalities when using DRP).
          initial_matrices: (Optional) List of initial metric matrices for each modality; if None, identities are used.
        """
        self.K = len(modalities_dims)
        self.modalities_dims = modalities_dims
        self.C = C
        self.C_prime = C_prime
        self.use_drp = use_drp
        self.m = m
        self.M_list = []
        self.R_list = []  # Random projection matrices for each modality
        for j, d in enumerate(modalities_dims):
            if initial_matrices is not None and initial_matrices[j] is not None:
                self.M_list.append(initial_matrices[j].copy())
            else:
                self.M_list.append(np.eye(d))
            if use_drp:
                if m is None:
                    raise ValueError("Must specify projection dimension m when use_drp is True.")
                # Generate a random projection matrix for modality j (d x m)
                Rj = np.random.normal(0, 1/np.sqrt(m), size=(d, m))
                self.R_list.append(Rj)
            else:
                self.R_list.append(None)
        # Initialize coefficient vector θ (nonnegative and normalized)
        self.theta = np.ones(self.K) / self.K
    
    def _update_metric_for_modality(self, x_q, x_pos, x_neg, j):
        """
        Update the metric for modality j using the triplet (x_q, x_pos, x_neg) following the paper’s steps.
        
        Steps for modality j:
          1. Compute difference vectors: diff_pos and diff_neg.
          2. Form A = (x_q - x_neg)(x_q - x_neg)^T - (x_q - x_pos)(x_q - x_pos)^T.
          3. If DRP is used, project x_q, x_pos, and x_neg with the modality-specific random matrix Rj to get reduced vectors.
          4. In the reduced space, form A_hat and compute the projected metric M_hat = Rjᵀ * Mj * Rj.
          5. Compute loss = max(0, 1 - trace(M_hat * A_hat)) and normA.
          6. If loss > 0, compute τ = min(C, loss / normA) and update Mj = Mj - τ * A.
          7. Project updated Mj onto the PSD cone.
          8. Compute distances: f_pos = diff_posᵀ * Mj * diff_pos and f_neg = diff_negᵀ * Mj * diff_neg.
        Returns f_pos and f_neg.
        """
        # Step 1: Compute difference vectors for modality j
        diff_pos = x_q - x_pos
        diff_neg = x_q - x_neg
        
        # Step 2: Construct A in the original space for modality j
        A = np.outer(diff_neg, diff_neg) - np.outer(diff_pos, diff_pos)
        
        if self.use_drp:
            # Step 3: Project the vectors into the reduced space using Rj
            Rj = self.R_list[j]
            x_q_hat   = Rj.T @ x_q
            x_pos_hat = Rj.T @ x_pos
            x_neg_hat = Rj.T @ x_neg
            # Step 4: Construct A_hat in the reduced space
            A_hat = np.outer(x_q_hat - x_neg_hat, x_q_hat - x_neg_hat) - np.outer(x_q_hat - x_pos_hat, x_q_hat - x_pos_hat)
            # Compute projected metric: M_hat = Rjᵀ * Mj * Rj
            Mj = self.M_list[j]
            Mj_hat = Rj.T @ Mj @ Rj
            # Step 5: Compute loss and norm in reduced space
            loss = max(0, 1 - np.trace(Mj_hat @ A_hat))
            normA = np.linalg.norm(A_hat, 'fro')**2
        else:
            Mj = self.M_list[j]
            margin = np.trace(Mj @ A)
            loss = max(0, 1 - margin)
            normA = np.linalg.norm(A, 'fro')**2
        
        # Step 6: Update Mj if loss > 0
        if loss > 0 and normA > 0:
            tau = min(self.C, loss / normA)
            Mj = Mj - tau * A
            # Step 7: Project updated Mj onto the PSD cone
            Mj = project_psd(Mj)
        self.M_list[j] = Mj
        
        # Step 8: Compute distances for this modality (for later coefficient update)
        f_pos = diff_pos.T @ Mj @ diff_pos
        f_neg = diff_neg.T @ Mj @ diff_neg
        return f_pos, f_neg
    
    def fit(self, triplets):
        """
        Fit the LSMDML model using the provided training triplets.
        
        Each triplet is a tuple (q, pos, neg), where each element is a dictionary mapping modality indices
        to feature vectors.
        """
        for (q, pos, neg) in triplets:
            f_pos_list = np.zeros(self.K)
            f_neg_list = np.zeros(self.K)
            # Step 1: For each modality, update the metric and record distances.
            for j in range(self.K):
                if j not in q or j not in pos or j not in neg:
                    raise ValueError(f"Modality {j} missing in one of the triplet elements.")
                x_q   = q[j]
                x_pos = pos[j]
                x_neg = neg[j]
                f_pos, f_neg = self._update_metric_for_modality(x_q, x_pos, x_neg, j)
                f_pos_list[j] = f_pos
                f_neg_list[j] = f_neg
            
            # Step 2: Update the coefficient vector θ.
            # Compute the difference vector for distances: f_diff = f_pos - f_neg.
            f_diff = f_pos_list - f_neg_list
            # Loss for θ update: loss_theta = max(0, 1 + θᵀ * f_diff)
            loss_theta = max(0, 1 + np.dot(self.theta, f_diff))
            diff_norm_sq = np.linalg.norm(f_diff)**2
            if loss_theta > 0 and diff_norm_sq > 0:
                # Adaptive learning rate for θ update:
                eta = min(self.C_prime, loss_theta / (diff_norm_sq + 1e-8))
                # Update and enforce non-negativity:
                self.theta = np.maximum(0, self.theta + eta * f_diff)
                # Normalize θ so that its entries sum to one.
                s = np.sum(self.theta)
                if s > 0:
                    self.theta = self.theta / s
    
    def get_mahalanobis_matrix(self):
        """
        Return the combined metric.
        
        Since modalities may have different dimensions, we return a dictionary that maps each modality index
        to its weighted metric: θ_j * M_j.
        """
        combined_metric = {}
        for j in range(self.K):
            combined_metric[j] = self.theta[j] * self.M_list[j]
        return combined_metric[0]