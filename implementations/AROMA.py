import numpy as np

class FactoredAROMA:
    """
    Factored AROMA for metric learning (based on:
    'Adaptive Regularization of Matrix Models (AROMA)'
     by Crammer and Chechik et al.)

    This class implements the update rules exactly as in 
    Equations (12), (13), and (14) from the paper, including 
    the denominators that ensure the PSD updates.
    """

    def __init__(self, r=0.01, m=None, n=None, initial_W=None):
        """
        Initialize the Factored AROMA model.

        Parameters:
            r (float): Regularization parameter.
            m (int): Dimensionality of query vectors.
            n (int): Dimensionality of candidate vectors.
            initial_W (np.ndarray): Optional initial weight matrix of shape (m, n).
                                     If provided, m and n are inferred from it.
        """
        self.r = r
        if initial_W is not None:
            self.W = initial_W.astype(np.float64)
            self.m, self.n = self.W.shape
        else:
            if m is None or n is None:
                raise ValueError("Either provide initial_W or specify both m and n")
            self.m, self.n = m, n
            self.W = np.zeros((self.m, self.n), dtype=np.float64)
        
        # Initialize the factorized covariance matrices (both PSD).
        # Lambda corresponds to the query side (m x m) 
        # Omega corresponds to the candidate side (n x n)
        self.Lambda = np.eye(self.m, dtype=np.float64)
        self.Omega = np.eye(self.n, dtype=np.float64)

    def fit(self, pairs, labels, epochs=1):
        """
        Train the model. The training data is assumed to be organized as follows:
          - pairs: a list of tuples (query, candidate)
          - labels: a list of labels (1 for similar, -1 for dissimilar)

        The model groups every two pairs (and labels) that share the same query 
        to form a triplet:
            1) (q, p_plus) with label = 1
            2) (q, p_minus) with label = -1

        Then it updates according to eq. (12), (13), and (14) from the paper.
        """
        if len(pairs) % 2 != 0 or len(labels) % 2 != 0:
            raise ValueError("Number of pairs and labels must be even (one similar + one dissimilar per query).")
        
        num_triplets = len(pairs) // 2

        for epoch in range(epochs):
            for t in range(num_triplets):
                idx1 = 2 * t
                idx2 = idx1 + 1
                
                # Extract the triplet
                q1, cand1 = pairs[idx1]
                q2, cand2 = pairs[idx2]
                label1 = labels[idx1]
                label2 = labels[idx2]
                
                # Basic checks
                if not np.allclose(q1, q2):
                    raise ValueError("The two pairs for a triplet must have the same query.")
                if label1 != 1 or label2 != -1:
                    raise ValueError("Expected labels are 1 (similar) and -1 (dissimilar).")
                
                # Convert to numpy arrays
                q = np.array(q1, dtype=np.float64).reshape(-1)   # shape (m,)
                p_plus = np.array(cand1, dtype=np.float64).reshape(-1)  # shape (n,)
                p_minus = np.array(cand2, dtype=np.float64).reshape(-1) # shape (n,)
                
                # Form the difference vector p = p_plus - p_minus
                p = p_plus - p_minus
                
                # Compute current similarity score: q^T * W * p
                score = q.dot(self.W).dot(p)
                
                # Hinge loss margin
                margin = 1.0 - score
                if margin > 0:
                    # Equation (12) from the paper:
                    #    W_i = W_{i-1} + [max(0, 1 - q^T W_{i-1} p)] / 
                    #                [r + (q^T Lambda_{i-1} q) (p^T Omega_{i-1} p)]
                    #            * (Lambda_{i-1} q) (p^T Omega_{i-1})^T
                    #
                    # We'll call the numerator "loss" and the denominator "denomW"
                    loss = margin
                    qt_L_q = q.dot(self.Lambda).dot(q)  # q^T Lambda q
                    pt_O_p = p.dot(self.Omega).dot(p)   # p^T Omega p
                    denomW = self.r + qt_L_q * pt_O_p
                    
                    # Outer product part: (Lambda q) (Omega p)^T
                    Lq = self.Lambda.dot(q)    # shape (m,)
                    Op = self.Omega.dot(p)     # shape (n,)
                    
                    self.W += (loss / denomW) * np.outer(Lq, Op)
                    
                    # Equation (13) from the paper:
                    #    Omega_i = Omega_{i-1} 
                    #             - [ (q^T Lambda_{i-1} q ) / (m*r + q^T Lambda_{i-1} q) ] 
                    #               * (Omega_{i-1} p) (Omega_{i-1} p)^T
                    #
                    # We'll call the factor for Omega "factorOmega"
                    denomOmega = (self.m * self.r) + qt_L_q * pt_O_p
                    factorOmega = qt_L_q / denomOmega
                    
                    # (Omega p) is shape (n,)
                    Op = self.Omega.dot(p)  # shape (n,)
                    # Outer product (Omega p) (Omega p)^T
                    Op_OpT = np.outer(Op, Op)
                    self.Omega -= factorOmega * (Op_OpT)
                    
                    # Equation (14) from the paper:
                    #    Lambda_i = Lambda_{i-1} 
                    #             - [ (p^T Omega_{i-1} p ) / (n*r + p^T Omega_{i-1} p) ] 
                    #               * (Lambda_{i-1} q) (Lambda_{i-1} q)^T
                    #
                    # We'll call the factor for Lambda "factorLambda"
                    pt_O_p = p.dot(self.Omega).dot(p)  # re-calc if needed
                    denomLambda = (self.n * self.r) + pt_O_p * qt_L_q
                    factorLambda = pt_O_p / denomLambda
                    
                    Lq = self.Lambda.dot(q)   # shape (m,)
                    Lq_LqT = np.outer(Lq, Lq)
                    self.Lambda -= factorLambda * (Lq_LqT)

    def predict(self, query, candidate):
        """
        Given a query and a candidate, compute the similarity score q^T W candidate.
        """
        q = np.array(query, dtype=np.float64).reshape(-1)
        cand = np.array(candidate, dtype=np.float64).reshape(-1)
        return q.dot(self.W).dot(cand)

    def get_mahalanobis_matrix(self):
        """
        Return the learned weight matrix W.
        (Kept exactly as in original code)
        """
        return self.W