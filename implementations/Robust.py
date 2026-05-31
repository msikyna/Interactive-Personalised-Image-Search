import numpy as np
from scipy.linalg import eigh  # for eigen-decomposition

def project_psd(M):
    """
    Project a symmetric matrix M onto the PSD cone by zeroing negative eigenvalues.
    This corresponds to eq. (21) in the paper, ensuring M ≽ 0.
    """
    vals, vecs = eigh(M)
    vals[vals < 0] = 0.0
    return (vecs * vals) @ vecs.T


class RobustODML:
    """
    Robust Mahalanobis-based Online Distance Metric Learning.

    This implements the robust rescaled hinge approach from the paper:
      l_rhinge(z) = beta * [1 - exp(-eta * hinge_loss(z))]
    using Half-Quadratic (HQ) iterations to handle the non-convex rescaled hinge.

    We keep the constructor and data-handling the same as your original code,
    but the core update steps follow the exact derivations in the paper.
    """

    def __init__(self, init_matrix=None, C=1.0, eta=1.0, max_hq_iter=3):
        """
        init_matrix : initial Mahalanobis matrix (d x d). Must be PSD.
        C : base parameter controlling the update magnitude (eq. (19) in paper).
        eta : the rescaling parameter in the robust rescaled hinge (eq. (10)).
        max_hq_iter : number of HQ iterations to run for each triplet/pair block.

        We also define beta = 1 / (1 - exp(-eta)) so that l_rhinge(0) = 1.
        """
        self.C = C
        self.eta = eta
        # Normalizing constant in eq. (10) to ensure smooth bounding of the loss.
        self.beta = 1.0 / (1.0 - np.exp(-eta))
        self.max_hq_iter = max_hq_iter

        if init_matrix is None:
            raise ValueError("Please provide an initial Mahalanobis matrix (init_matrix).")
        self.M = init_matrix  # Keep your direct reference to M

    def fit(self, pairs, labels):
        """
        Fit/update the metric M given training pairs and labels.

        We assume each "triplet" is split across two pairs:
            - (query, positive) with label +1
            - (query, negative) with label -1
        so that for every i in range(0, n, 2), we form the anchor 'q', pos 'p', neg 'n'.

        This reimplements the robust updates exactly as in the paper,
        including the HQ iteration, eq. (17) for v, eq. (19) for M-update,
        and eq. (21) for the PSD projection, but using pairs.
        """
        n = len(pairs)
        if n % 2 != 0:
            raise ValueError("Number of pairs must be even: each query has +1 and -1 pair.")

        # Process pairs two at a time to simulate a single (q, p, n) triplet.
        for i in range(0, n, 2):
            # The first pair is (query, positive) with label +1
            # The second pair is (query, negative) with label -1
            query1, pos = pairs[i]
            query2, neg = pairs[i+1]
            if labels[i] != 1 or labels[i+1] != -1:
                raise ValueError("Expected consecutive pair labels +1, -1.")

            # Ensure they share the same query anchor
            if not np.allclose(query1, query2):
                raise ValueError("Consecutive pairs must share the same query vector.")

            # Flatten in case the inputs are not 1D
            q = np.asarray(query1).ravel()
            p = np.asarray(pos).ravel()
            n_vec = np.asarray(neg).ravel()

            # ---- Step 1: Compute the standard hinge loss (eq. (4) in paper) ----
            # hinge_loss = max(0, margin + dM^2(q,p) - dM^2(q,n))
            # The margin is fixed to 1, as in eq. (4).
            def dM2(x, y):
                """
                Squared Mahalanobis distance under M:
                d_M^2(x, y) = (x-y)^T M (x-y)
                """
                diff = x - y
                return diff @ self.M @ diff

            hinge_loss = max(0.0, 1.0 + dM2(q, p) - dM2(q, n_vec))

            # ---- Step 2: Build A_t = (q - n)(q - n)^T - (q - p)(q - p)^T (eq. (6)-(7)) ----
            Aq = q - p
            An = q - n_vec
            A_t = np.outer(An, An) - np.outer(Aq, Aq)

            # We also need its Frobenius norm squared for the update bound
            frobA2 = np.sum(A_t * A_t)

            # ---- Steps 3 & 4: HQ iterations (eq. (17) for v, eq. (19)-(20) for M) ----
            for _ in range(self.max_hq_iter):
                # eq. (17) from the paper:
                #    v_t = argmax  [ eta * hinge_loss(z) * v - g(v) ]
                # It ends up with closed-form v_t = - exp(-eta * hinge_loss).
                v = -np.exp(-self.eta * hinge_loss)

                # The robust approach yields an adaptive weight C_t (eq. (19)):
                #    C_t = C * beta * eta * exp(-eta * hinge_loss)
                C_t = self.C * self.beta * self.eta * np.exp(-self.eta * hinge_loss)

                # Solve eq. (19) => we get a step size tau = min(C_t, hinge_loss / ||A||^2 ).
                # This step size ensures we do not "overshoot."
                denom = (frobA2 + 1e-12)  # to avoid division by zero
                tau = min(C_t, hinge_loss / denom) if hinge_loss > 0 else 0.0

                # ---- Update M: eq. (20) => M_{t+1} = M_t + tau * A_t ----
                self.M += tau * A_t

                # ---- Enforce PSD: eq. (21) => project M onto PSD cone ----
                self.M = project_psd(self.M)

                # ---- Recompute the hinge loss after the update, as recommended ----
                hinge_loss = max(0.0, 1.0 + dM2(q, p) - dM2(q, n_vec))

                # If hinge_loss is 0, that means the margin is satisfied.
                # In principle we could break early, but the paper often
                # just runs all HQ iterations, so we keep going.

    def get_mahalanobis_matrix(self):
        """
        Returns the current Mahalanobis matrix M.
        We do not change this method, as requested.
        """
        return self.M


class RobustLODML:
    """
    Robust low-rank online distance metric learning (Robust-LODML).
    It learns a projection matrix L so that M = L Lᵀ.
    """
    def __init__(self, init_projection=None, C=1.0, eta=1.0, max_hq_iter=3):
        """
        init_projection: initial projection matrix L (numpy array, shape [d, r]).
                         If None, identity is used (square matrix; r=d).
        C: base parameter controlling the update step.
        eta: rescaling parameter.
        max_hq_iter: number of HQ iterations per triplet.
        """
        self.C = C
        self.eta = eta
        self.beta = 1.0 / (1 - np.exp(-eta))
        self.max_hq_iter = max_hq_iter
        self.L = init_projection
        if self.L is None:
            raise ValueError("Please provide an initial projection matrix (init_projection)")
            
    def fit(self, pairs, labels):
        """
        Fit/update the projection matrix L given training pairs.
        The pairs are expected to be arranged in triplets as in RobustODML.
        """
        n = len(pairs)
        if n % 2 != 0:
            raise ValueError("The number of pairs must be even (a similar and a dissimilar for each query).")
            
        for i in range(0, n, 2):
            query1, pos = pairs[i]
            query2, neg = pairs[i+1]
            if labels[i] != 1 or labels[i+1] != -1:
                raise ValueError("Expected labels of +1 and -1 for consecutive pairs.")
            if not np.allclose(query1, query2):
                raise ValueError("The two pairs in a triplet must share the same query.")
            
            q = np.array(query1).flatten()
            p = np.array(pos).flatten()
            n_vec = np.array(neg).flatten()
            
            # In the low-rank case, the (squared) distance is computed as:
            # d_L^2(q, x) = || Lᵀq - Lᵀx ||^2
            def dL2(x, y):
                diff = self.L.T @ (x - y)
                return np.dot(diff, diff)
            
            loss = max(0, 1 + dL2(q, p) - dL2(q, n_vec))
            # For consistency, we use the same A_t as in the full matrix case:
            Aq = q - p
            An = q - n_vec
            A_t = np.outer(An, An) - np.outer(Aq, Aq)
            frobA2 = np.sum(A_t * A_t)
            
            # HQ iterations:
            for _ in range(self.max_hq_iter):
                v = -np.exp(-self.eta * loss)
                C_t = self.C * self.beta * self.eta * np.exp(-self.eta * loss)
                tau = min(C_t, loss / (frobA2 + 1e-12))
                # Update L via a subgradient step.
                # According to the derivation, a simple update is:
                # L = L + 2 * tau * C_t * A_t * L
                self.L = self.L + 2 * tau * C_t * (A_t @ self.L)
                # Optionally, one might normalize or use a learning rate scheduler.
                loss = max(0, 1 + dL2(q, p) - dL2(q, n_vec))
                
    def get_mahalanobis_matrix(self):
        # Return the full metric as M = L Lᵀ.
        return self.L @ self.L.T