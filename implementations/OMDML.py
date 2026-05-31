import numpy as np
from numpy.linalg import eigh, norm

def psd_projection(M):
    """
    Project a symmetric matrix M onto the PSD cone.
    (This is used in Step 12 of Algorithm 1 to ensure M is PSD.)
    """
    # Compute eigen-decomposition
    eigvals, eigvecs = eigh(M)
    # Set negative eigenvalues to 0 (i.e., project to nonnegative eigenvalues)
    eigvals[eigvals < 0] = 0
    # Reconstruct the PSD matrix
    return (eigvecs * eigvals) @ eigvecs.T

def ensure_multimodal(x, m):
    """
    Ensure that x is represented as a list of m numpy arrays.
    If x is a single modality (not a list), replicate it m times.
    (This corresponds to the input processing before the learning phase.)
    """
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x] * m

class OMDML:
    """
    Online Multi-modal Distance Metric Learning (OMDML) class.
    
    This implementation follows exactly the paper’s Algorithm 1.
    
    Parameters:
      beta: Discount weight (0 < beta < 1) used in the Hedge update.
      C: Regularization parameter used for step-size calculation.
      gamma: Margin parameter.
      m: Number of modalities.
      prior: Optional list of initial Mahalanobis matrices (one per modality).
             If None, each matrix is set to the identity matrix when first used.
    """
    def __init__(self, beta=0.9, C=1.0, gamma=0.0, m=1, prior=None):
        self.beta = beta
        self.C = C
        self.gamma = gamma
        self.m = m
        # Step 2: Initialize combination weights uniformly
        self.theta = np.ones(m) / m
        # Set initial Mahalanobis matrices (to be initialized later if not provided)
        self.M = prior if prior is not None else [None] * m

    def _initialize_modalities(self, features):
        """
        For each modality, initialize the Mahalanobis matrix to the identity 
        if it has not been set yet.
        (This corresponds to Step 2 in the paper.)
        """
        for i in range(self.m):
            if self.M[i] is None:
                d = features[i].shape[0]
                self.M[i] = np.eye(d)

    def fit(self, pairs, labels, n_iter=None):
        """
        Train the model on a stream of triplet constraints.
        
        The input is assumed to be in ITML style:
          - For each query, two pairs are given consecutively: the first with label 1 (similar)
            and the second with label -1 (dissimilar).
        
        This method implements the following steps from the paper:
          4. Receive a triplet (p_t, p^+_t, p^-_t)
          5. For each modality i, compute f^(i)_t = d_i(p_t, p^+_t) - d_i(p_t, p^-_t)
          6. Compute f_t = sum_i θ(i)_t * f^(i)_t
          7. If f_t + γ > 0 then update:
              8-9. For each modality i, compute z(i)_t = I(f^(i)_t > 0)
             10. Update combination weights: θ(i)_{t+1} = θ(i)_t * beta^(z(i)_t)
             11. Update Mahalanobis matrix for each modality:
                     M(i)_{t+1} = M(i)_t - τ(i)_t * V(i)_t
                 where V(i)_t = outer(p_t - p^+_t, p_t - p^+_t) - outer(p_t - p^-_t, p_t - p^-_t)
             12. Project updated M(i) onto the PSD cone.
             14. Normalize the combination weights.
        """
        # Number of triplets (each triplet is made from two consecutive pairs)
        n_triplets = len(pairs) // 2
        for t in range(n_triplets):
            # === Step 4: Receive triplet ===
            # Each triplet is formed by two pairs:
            #   Pair1: (query, positive) with label 1
            #   Pair2: (query, negative) with label -1
            pair1 = pairs[2 * t]
            pair2 = pairs[2 * t + 1]
            label1 = labels[2 * t]
            label2 = labels[2 * t + 1]
            if label1 != 1 or label2 != -1:
                raise ValueError("Expected labels 1 and -1 for similar and dissimilar pairs.")
            query = pair1[0]
            positive = pair1[1]
            negative = pair2[1]
            
            # === Preprocess inputs (same as before) ===
            query_mod = ensure_multimodal(query, self.m)
            pos_mod   = ensure_multimodal(positive, self.m)
            neg_mod   = ensure_multimodal(negative, self.m)
            
            # Initialize Mahalanobis matrices if needed (Step 2)
            self._initialize_modalities(query_mod)
            
            # === Step 5: Compute per-modality differences ===
            # For each modality i, calculate f^(i)_t = d_i(query, positive) - d_i(query, negative)
            f_vals = np.zeros(self.m)
            for i in range(self.m):
                diff_pos = query_mod[i] - pos_mod[i]
                diff_neg = query_mod[i] - neg_mod[i]
                d_pos = diff_pos.T @ self.M[i] @ diff_pos
                d_neg = diff_neg.T @ self.M[i] @ diff_neg
                f_vals[i] = d_pos - d_neg  # f^(i)_t
              
            # === Step 6: Compute the overall score ===
            f_total = np.dot(self.theta, f_vals)  # f_t = sum_i θ(i)_t f^(i)_t
              
            # === Step 7: Check margin violation and update if needed ===
            if f_total + self.gamma > 0:
                # === Steps 8-9: Compute indicator for each modality ===
                # z(i)_t = 1 if f^(i)_t > 0, else 0.
                z = np.array([1.0 if f > 0 else 0.0 for f in f_vals])
                
                # === Step 10: Update combination weights using Hedge ===
                # θ(i)_{t+1} = θ(i)_t * beta^(z(i)_t)
                self.theta = self.theta * (self.beta ** z)
                
                # === Step 11: For each modality, update the Mahalanobis matrix ===
                for i in range(self.m):
                    # Compute hinge loss for modality i:
                    # ℓ_t = max(0, d(query, positive) - d(query, negative) + 1)
                    loss = max(0, f_vals[i] + 1)
                    # Compute V(i)_t = outer(query - positive) - outer(query - negative)
                    diff_pos = query_mod[i] - pos_mod[i]
                    diff_neg = query_mod[i] - neg_mod[i]
                    V = np.outer(diff_pos, diff_pos) - np.outer(diff_neg, diff_neg)
                    V_norm_sq = norm(V, 'fro')**2
                    # Compute step size τ(i)_t = min(C, loss / ||V||_F^2) [Step 11]
                    tau = min(self.C, loss / V_norm_sq) if V_norm_sq > 0 else 0
                    # Update the Mahalanobis matrix: M(i)_{t+1} = M(i)_t - τ(i)_t * V(i)_t
                    self.M[i] = self.M[i] - tau * V
                    # === Step 12: PSD projection ===
                    self.M[i] = psd_projection(self.M[i])
                
                # === Step 14: Normalize the combination weights ===
                theta_sum = np.sum(self.theta)
                if theta_sum > 0:
                    self.theta = self.theta / theta_sum

    def get_mahalanobis_matrix(self):
        """
        Return the Mahalanobis matrix.
        (This method is kept unchanged as per your request.)
        Here, it returns the metric of the first modality.
        """
        return self.M[0]


import numpy as np
from numpy.linalg import norm

def ensure_multimodal(x, m):
    """
    Ensures that x is represented as a list of m numpy arrays.
    If x is not a list, it is assumed to be a single modality and is replicated.
    """
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x] * m

class LOMDML:
    """
    Low-rank Online Multi-modal Distance Metric Learning (LOMDML) class.
    This follows Algorithm 2 in the paper exactly.
    
    Parameters:
      beta:  Discount weight (0 < beta < 1) for the Hedge update.
      gamma: Margin parameter γ ≥ 0.
      eta:   Learning rate for the online gradient descent update.
      m:     Number of modalities.
      r:     Rank (projected dimension) for each modality.
      prior: Optional list of initial W matrices (one per modality).
             If None, each W is initialized as an identity-like matrix.
    """
    def __init__(self, beta=0.9, gamma=0.0, eta=0.001, m=1, r=50, prior=None):
        # === Step 2: Initialization ===
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.m = m
        self.r = r
        # Initialize weights θ(i) = 1/m
        self.theta = np.ones(m) / m
        # Initialize low-rank matrices W(i)
        self.W = prior if prior is not None else [None] * m

    def _initialize_modalities(self, features):
        """
        For each modality, initialize W(i) if not already set.
        W(i) has shape (r, d) where d is the feature dimension for that modality.
        """
        for i in range(self.m):
            if self.W[i] is None:
                d = features[i].shape[0]
                # If r ≤ d, use the top r rows of an identity matrix (I_d).
                if self.r <= d:
                    self.W[i] = np.eye(self.r, d)
                else:
                    # If r > d, pad I_d with zeros to make it shape (r, d)
                    self.W[i] = np.pad(np.eye(d), ((0, self.r - d), (0, 0)), mode='constant')
    
    def fit(self, pairs, labels, n_iter=None):
        """
        Train the model using triplet constraints.
        """
        n_triplets = len(pairs) // 2
        
        for t in range(n_triplets):
            # === Step 4: Receive triplet ===
            pair1 = pairs[2 * t]     # (query, positive) with label +1
            pair2 = pairs[2 * t + 1] # (query, negative) with label -1
            label1 = labels[2 * t]
            label2 = labels[2 * t + 1]
            if label1 != 1 or label2 != -1:
                raise ValueError("Expected labels 1 and -1 for similar and dissimilar pairs.")
            
            # Extract query, positive, negative
            query = pair1[0]
            positive = pair1[1]
            negative = pair2[1]
            
            # Convert to multi-modal format
            query_mod = ensure_multimodal(query, self.m)
            pos_mod   = ensure_multimodal(positive, self.m)
            neg_mod   = ensure_multimodal(negative, self.m)
            
            # Initialize W(i) if needed
            self._initialize_modalities(query_mod)
            
            # === Step 5: Compute f^(i)_t for each modality ===
            f_vals = np.zeros(self.m)
            for i in range(self.m):
                diff_pos = query_mod[i] - pos_mod[i]
                diff_neg = query_mod[i] - neg_mod[i]
                d_pos = norm(self.W[i] @ diff_pos) ** 2
                d_neg = norm(self.W[i] @ diff_neg) ** 2
                f_vals[i] = d_pos - d_neg
            
            # === Step 6: Compute total f_t ===
            f_total = np.dot(self.theta, f_vals)
            
            # === Step 7: If margin is violated, update ===
            if f_total + self.gamma > 0:
                # === Step 8: Compute z^(i)_t ===
                z = (f_vals > 0).astype(float)
                
                # === Step 9: Update Hedge Weights ===
                self.theta *= (self.beta ** z)
                
                # === Step 10-12: Update W(i) using Online Gradient Descent ===
                for i in range(self.m):
                    W_i = self.W[i]
                    q       = W_i @ query_mod[i]
                    q_plus  = W_i @ pos_mod[i]
                    q_minus = W_i @ neg_mod[i]
                    
                    # Gradient update (Eq. 7 in the paper)
                    grad = (2 * np.outer(-q_plus + q_minus, query_mod[i]) +
                            2 * np.outer(-q + q_plus,       pos_mod[i])   +
                            2 * np.outer(q - q_minus,       neg_mod[i]))
                    
                    # **Step 12: Apply gradient update**
                    self.W[i] = W_i - self.eta * grad

                # === Step 14: Compute normalization constant for θ ===
                Theta_t1 = np.sum(self.theta)

                # === Step 15: Normalize θ(i)_{t+1} ===
                if Theta_t1 > 0:
                    self.theta /= Theta_t1

    def get_mahalanobis_matrix(self):
        """
        Return the learned Mahalanobis matrices: M(i) = W(i)^T * W(i).
        """
        M_list = []
        for i in range(self.m):
            M_i = self.W[i].T @ self.W[i]
            M_list.append(M_i)
        return M_list[0]