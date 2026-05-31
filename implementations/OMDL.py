import numpy as np
from numpy.linalg import eigh

def project_psd(M):
    """
    Projects a symmetric matrix M onto the PSD cone by zeroing out
    any negative eigenvalues.
    """
    # Compute eigen-decomposition (since M is symmetric)
    eigvals, eigvecs = eigh(M)
    eigvals[eigvals < 0] = 0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

class OMDL:
    def __init__(self, C1=1.0, C2=1.0, eta=0.9, prior=None):
        """
        Initialize the OMDL model.
        
        Parameters:
          C1: regularization trade-off parameter (not explicitly used here in a Laplacian term)
          C2: aggressiveness parameter (upper bound on the step size)
          eta: discount factor used in updating the combination weight mu
          prior: initial Mahalanobis matrix (if None, identity is used)
        """
        self.C1 = C1
        self.C2 = C2
        self.eta = eta
        self.M = prior  # Mahalanobis matrix
        self.mu = 1.0   # combination weight (for single–kernel case)
        self.initialized = False

    def _initialize(self, dim):
        """
        Initialize the Mahalanobis matrix to identity if no prior is given.
        """
        if self.M is None:
            self.M = np.eye(dim)
        self.initialized = True

    def fit(self, pairs, labels):
        """
        Fit the OMDL model on training data provided as pairs.
        
        The training data are assumed to be provided as follows:
          - pairs: list of tuples, each tuple is (query_feature, file_feature)
          - labels: list of labels: +1 for a similar pair and -1 for a dissimilar pair.
          
        It is assumed that every two consecutive pairs form a triplet:
          (q, s) with label +1 and (q, d) with label -1,
        corresponding to the triplet (q, s, d).
        """
        num_triplets = len(pairs) // 2
        for t in range(num_triplets):
            # Extract triplet: query, similar, dissimilar
            q, sim = pairs[2 * t]
            _, dissim = pairs[2 * t + 1]

            # Initialize M based on feature dimension if not already set
            if not self.initialized:
                dim = q.shape[0]
                self._initialize(dim)

            # Compute difference vectors
            diff_pos = q - sim   # positive difference
            diff_neg = q - dissim  # negative difference

            # Compute current distances
            d_pos = diff_pos.T @ self.M @ diff_pos
            d_neg = diff_neg.T @ self.M @ diff_neg

            # Compute hinge loss: loss = d_pos - d_neg + 1
            loss = d_pos - d_neg + 1

            if loss > 0:
                # Compute gradient matrix G = (q-s)(q-s)^T - (q-d)(q-d)^T
                G = np.outer(diff_pos, diff_pos) - np.outer(diff_neg, diff_neg)
                # Frobenius norm squared of G
                G_norm_sq = np.sum(G ** 2)
                if G_norm_sq == 0:
                    tau = 0
                else:
                    tau = min(self.C2, loss / G_norm_sq)
                # Update the Mahalanobis matrix
                self.M = self.M - tau * G
                # Project onto PSD cone
                self.M = project_psd(self.M)
            
            # Update the combination weight mu using a hedging update.
            # If the positive distance exceeds the negative distance, set indicator = 1.
            indicator = 1 if d_pos > d_neg else 0
            self.mu = self.mu * (self.eta ** indicator)

    def get_mahalanobis_matrix(self):
        """
        Returns the current learned Mahalanobis matrix.
        For the single–kernel case, the final matrix is mu * M.
        """
        return self.mu * self.M