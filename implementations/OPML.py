import numpy as np
from scipy.linalg import cholesky

class OPML:
    """
    One-Pass Metric Learning (OPML) implementation that directly applies
    Equation (16) in the paper for the update:

        L_t = L_{t-1}
               - (eta * gamma)/(eta+beta) * [L_{t-1} a a^T - L_{t-1} b b^T]
               + (gamma^2)/(eta+beta) * [
                    (a^T a) * L_{t-1} a a^T
                  - (a^T b) * L_{t-1} a b^T
                  - (b^T a) * L_{t-1} b a^T
                  + (b^T b) * L_{t-1} b b^T
                 ]

    The hinge loss is:
        loss = max(0, 1 + ||L a||^2 - ||L b||^2).

    If loss > 0, we apply the above closed-form update. 
    Otherwise, L remains unchanged for that triplet.

    Parameters
    ----------
    gamma : float
        Regularization/learning rate parameter (0 < gamma < 1/4 for theoretical 
        guarantees in the paper).
    initial_matrix : None or np.ndarray
        If None, L is set to identity on the first triplet. Otherwise,
        pass a d×d ndarray as the initial transformation matrix.

    Attributes
    ----------
    L : np.ndarray
        The current d×d transformation matrix. The Mahalanobis matrix is M = LᵀL.
    """

    def __init__(self, gamma=1e-3, initial_matrix=None):
        self.gamma = gamma
        self.L = self.decompose(initial_matrix)  # Will initialize to identity on first use if None
    
    def decompose(self, matrix):
        """
        Decomposes M into L such that M = L^T L (Cholesky factor).
        If small numerical issues arise, add a tiny eps on diagonal. 
        This is the step "Then, M is decomposed as L^T L" from the paper's text.
        """
        try:
            L = cholesky(matrix, lower=True)
        except np.linalg.LinAlgError:
            eps = 1e-8
            L = cholesky(matrix + eps*np.eye(matrix.shape[0]), lower=True)
        return L

    def fit(self, triplets):
        """
        Update the transformation matrix L by iterating over the provided triplets.

        Parameters
        ----------
        triplets : list of (x, x_p, x_q)
            Each element is (query, positive, negative), where:
              - x   = query sample
              - x_p = positive sample (same class as x)
              - x_q = negative sample (different class from x)

        Returns
        -------
        self
        """
        for (x, x_p, x_q) in triplets:
            x   = np.asarray(x).ravel()
            x_p = np.asarray(x_p).ravel()
            x_q = np.asarray(x_q).ravel()

            # Initialize L if not set yet
            if self.L is None:
                d = x.shape[0]
                self.L = np.eye(d)

            # Compute a, b
            a = x - x_p
            b = x - x_q

            # Hinge-loss check: 1 + ||L a||^2 - ||L b||^2
            La = self.L.dot(a)
            Lb = self.L.dot(b)
            loss_val = 1.0 + np.dot(La, La) - np.dot(Lb, Lb)
            if loss_val <= 0:
                # No update if the margin is already satisfied
                continue

            # Otherwise apply the update of Eq.(16).
            # 1) Precompute basic dot-products
            aTa = a @ a  # a^T a
            aTb = a @ b  # a^T b
            bTa = b @ a  # b^T a (same as aTb if real)
            bTb = b @ b  # b^T b

            # 2) We need gamma * A = gamma * (a a^T - b b^T)
            #    Then compute trace(gammaA) = gamma*(aTa - bTb), etc.
            trA = self.gamma * (aTa - bTb)     # trace( gamma * (a a^T - b b^T) )
            eta = 1.0 + trA                    # eta = 1 + trace(gammaA)

            # 3) Compute (gammaA)^2 and its trace to get beta
            #    We can do rank-2 multiplication in O(d^2).
            #    But let's do a direct approach:
            gammaA = self.gamma * (np.outer(a, a) - np.outer(b, b))
            gammaA2 = gammaA @ gammaA
            trace_gammaA2 = np.trace(gammaA2)
            beta = 0.5 * (trA**2 - trace_gammaA2)

            denom = eta + beta
            if abs(denom) < 1e-14:
                # Degenerate fallback: do a normal inverse-based update
                # to prevent divide-by-zero.
                I = np.eye(self.L.shape[0])
                inv_mat = np.linalg.inv(I + gammaA)
                self.L = self.L.dot(inv_mat)
                continue

            # 4) Build each component of Eq.(16)
            #    We'll do L_{t-1} a a^T, L_{t-1} b b^T, etc.
            Lt_aa = self.L @ np.outer(a, a)
            Lt_bb = self.L @ np.outer(b, b)
            Lt_ab = self.L @ np.outer(a, b)
            Lt_ba = self.L @ np.outer(b, a)

            # The first bracket: - (eta * gamma) / (eta + beta) * [ L a a^T - L b b^T ]
            coef1 = -(eta * self.gamma) / denom
            bracket1 = coef1 * (Lt_aa - Lt_bb)

            # The second bracket: (gamma^2)/(eta+beta)* [ (aTa)L a a^T - (aTb)L a b^T
            #                                             - (bTa)L b a^T + (bTb)L b b^T ]
            coef2 = (self.gamma**2) / denom
            bracket2 = ( aTa * Lt_aa
                       - aTb * Lt_ab
                       - bTa * Lt_ba
                       + bTb * Lt_bb )
            bracket2 *= coef2

            # Final update: L_t = L_{t-1} + bracket1 + bracket2
            self.L = self.L + bracket1 + bracket2

        return self

    def get_mahalanobis_matrix(self):
        """
        Returns the Mahalanobis matrix M = Lᵀ L.
        """
        return self.L.T @ self.L