import numpy as np
from scipy.linalg import svd

class LORETA:
    """
    A corrected implementation of LORETA (for PSD matrices M=L Lᵀ) that follows 
    the paper "Online Learning in the Manifold of Low-Rank Matrices" (NIPS 2010).

    The update here performs:
      1. A Euclidean gradient step w.r.t. L,
      2. Projection to the tangent space (via M_sym and S),
      3. The second-order retraction (Theorem 1) for PSD manifolds,
      4. Rank-k truncation via an eigen-decomposition of new_x.

    Note: We explicitly symmetrize new_x before the EVD to ensure numerical stability.
    """

    def __init__(self, n_features, rank=10, learning_rate=1e-3, margin=1.0, 
                 initial_M=None, random_state=None):
        self.n_features = n_features
        self.rank = rank
        self.learning_rate = learning_rate
        self.margin = margin
        self.random_state = random_state
        self.initial_M = initial_M
        self._initialize_model()

    def _initialize_model(self):
        rng = np.random.RandomState(self.random_state)
        if self.initial_M is not None:
            if self.initial_M.shape != (self.n_features, self.n_features):
                raise ValueError("initial_M must be of shape (n_features, n_features)")
            # Ensure symmetry
            if not np.allclose(self.initial_M, self.initial_M.T, atol=1e-6):
                raise ValueError("initial_M must be symmetric")
            # Use eigen-decomposition and keep the top 'rank' components
            eigenvalues, eigenvectors = np.linalg.eigh(self.initial_M)
            idx = np.argsort(eigenvalues)[::-1]
            top_eigenvalues = eigenvalues[idx][:self.rank]
            top_eigenvectors = eigenvectors[:, idx][:, :self.rank]
            # Build L so that M = L Lᵀ
            self.L = top_eigenvectors * np.sqrt(np.maximum(top_eigenvalues, 0))
        else:
            # Random low-scale initialization
            self.L = rng.randn(self.n_features, self.rank) * 0.01

    def _orthogonal_complement(self, A):
        """
        Use QR decomposition to compute an orthonormal basis for the orthogonal 
        complement of A's columns.
        """
        Q, R = np.linalg.qr(A, mode='complete')
        # The last columns of Q form an orthonormal basis orthogonal to A
        return Q[:, A.shape[1]:]

    def _safe_pinv(self, A, eps=1e-8, max_iter=5):
        """
        Compute the pseudoinverse of A with added regularization.
        If the SVD or solve does not converge, increase the regularization term.
        """
        factor = 1
        while factor <= 10**max_iter:
            try:
                return np.linalg.pinv(A + factor * eps * np.eye(A.shape[0]))
            except np.linalg.LinAlgError:
                factor *= 10
        return np.linalg.pinv(A + factor * eps * np.eye(A.shape[0]))

    def _retraction_theorem(self, dL):
        """
        Retract using the exact second-order retraction operator from Theorem 1 
        (adapted to the symmetric PSD manifold x = L Lᵀ).

        dL is the update for L (the tangent step), typically = -eta * Euclidean_grad(L).
        Returns new_x = w1 @ x_dagger @ w2, the retracted M on the manifold.
        """
        L = self.L
        x = L @ L.T  # current point M on the manifold
        eps = 1e-8   # base regularization for safe_pinv

        # Compute the pseudoinverse of L: L_dagger = (Lᵀ L)⁻¹ Lᵀ
        LT_L = L.T @ L  # shape (rank x rank)
        L_dagger = self._safe_pinv(LT_L, eps=eps, max_iter=5) @ L.T

        # (1) Project dL onto the tangent space. 
        #     M = L_dagger * dL, then symmetrize => M_sym.  
        M = L_dagger @ dL            # shape (rank x rank)
        M_sym = 0.5 * (M + M.T)
        S = dL - L @ M_sym           # part of dL orthogonal to L

        # (2) Build the orthonormal complement L_perp
        L_perp = self._orthogonal_complement(L)  # shape (n_features x (n_features - rank))

        # (3) Represent S in basis L_perp => N
        N = L_perp.T @ S            # shape ((n_features - rank) x rank)

        # (4) Decompose tangent vector ξ into xi_S, xi_p_l, xi_p_r
        xi_S   = L        @ M_sym @ L.T
        xi_p_l = L_perp   @ N      @ L.T
        xi_p_r = L        @ N.T    @ L_perp.T

        # (5) Pseudoinverse of x
        x_dagger = self._safe_pinv(x, eps=eps, max_iter=5)

        # (6) w1, w2 from Theorem 1
        w1 = x + 0.5 * xi_S + xi_p_r \
             - 0.125 * (xi_S @ x_dagger @ xi_S) \
             - 0.5   * (xi_p_r @ x_dagger @ xi_S)

        w2 = x + 0.5 * xi_S + xi_p_l \
             - 0.125 * (xi_S @ x_dagger @ xi_S) \
             - 0.5   * (xi_S @ x_dagger @ xi_p_l)

        # (7) Second-order retraction R_x(dL) = w1 * x_dagger * w2
        new_x = w1 @ x_dagger @ w2

        # (8) IMPORTANT: ensure numerical symmetry before EVD
        new_x = 0.5 * (new_x + new_x.T)

        return new_x

    def fit(self, pairs, epochs=1, verbose=False):
        """
        Perform online training using pairs of examples in "metric learning" style:
          pairs[0], pairs[1] => (query, positive)
          pairs[2], pairs[3] => (query, negative)
        with a hinge loss margin.

        For each match (q,pos) and mismatch (q,neg),
          hinge loss = max(0, margin + d_pos - d_neg)
          where d(x,y) = (Lᵀ (x-y))² is the Mahalanobis distance for M = L Lᵀ.
        """
        n_pairs = len(pairs)
        if n_pairs % 2 != 0:
            raise ValueError("Number of pairs must be even (each query must have both a positive and negative).")
    
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, n_pairs, 2):
                q1, pos = pairs[i]
                q2, neg = pairs[i+1]
                if not np.allclose(q1, q2, atol=1e-6):
                    raise ValueError("Consecutive pairs must share the same query.")
                q = q1  # common query

                # Compute differences
                diff_pos = q - pos
                diff_neg = q - neg
            
                # Squared Mahalanobis distances: d(x,y) = ||Lᵀ(x-y)||²
                d_pos = np.sum((self.L.T.dot(diff_pos))**2)
                d_neg = np.sum((self.L.T.dot(diff_neg))**2)
            
                # Hinge loss
                loss = max(0, self.margin + d_pos - d_neg)
                total_loss += loss

                if loss > 0:
                    # Euclidean gradient w.r.t. L for d_pos - d_neg
                    # grad_L = 2 * [ (diff_pos * diff_posᵀ) - (diff_neg * diff_negᵀ) ] * L
                    grad_L = 2.0 * (
                        np.outer(diff_pos, diff_pos) - np.outer(diff_neg, diff_neg)
                    ) @ self.L

                    # Step on negative gradient
                    dL = -self.learning_rate * grad_L

                    # Retract using second-order operator
                    new_x = self._retraction_theorem(dL)

                    # Regularize slightly for numerical stability, then keep top 'rank' eigen-components
                    reg = 1e-8
                    new_x_reg = new_x + reg * np.eye(new_x.shape[0])
                    eigenvalues, eigenvectors = np.linalg.eigh(new_x_reg)

                    # Sort eigenvalues descending
                    idx = np.argsort(eigenvalues)[::-1]
                    top_vals = eigenvalues[idx][:self.rank]
                    top_vecs = eigenvectors[:, idx][:, :self.rank]

                    # Update L so that M = L Lᵀ is the new matrix
                    self.L = top_vecs * np.sqrt(np.maximum(top_vals, 1e-6))

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss}")

    def get_mahalanobis_matrix(self):
        """
        Returns the learned Mahalanobis matrix M = L Lᵀ.
        """
        return self.L @ self.L.T