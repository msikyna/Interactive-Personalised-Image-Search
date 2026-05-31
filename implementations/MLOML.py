import numpy as np
from scipy.linalg import cholesky

##############################################################################
# Activation functions (unchanged).
##############################################################################
def relu(x):
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation: 1 / (1 + e^(-x))."""
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    """Hyperbolic tangent activation: (e^x - e^-x)/(e^x + e^-x)."""
    return np.tanh(x)

##############################################################################
# Project a matrix onto the PSD cone. (Paper's Theorem 1 notes ensuring M >= 0)
##############################################################################
def project_psd(M):
    """
    Projects matrix M onto PSD by eigen-decomposition, clipping negative eigenvalues to 0,
    and reconstructing. Ensures M remains positive semidefinite.
    """
    # Symmetrize for numerical stability
    M_sym = 0.5 * (M + M.T)
    e, V = np.linalg.eigh(M_sym)
    e[e < 0.0] = 0.0
    M_psd = (V @ np.diag(e) @ V.T)
    return M_psd

##############################################################################
# Single-layer Mahalanobis OML: MOMLLayer (Paper, Section III-A).
##############################################################################
class MOMLLayer:
    def __init__(self, d, gamma=0.01, M_init=None):
        """
        d      : dimensionality of input features.
        gamma  : learning rate (aggressiveness parameter) for the MOML update.
        M_init : optional initial Mahalanobis matrix (shape d x d); if None, use identity.
        """
        self.d = d
        self.gamma = gamma
        self.M = np.eye(d) if M_init is None else M_init.copy()

    def local_update(self, x, x_pos, x_neg, margin=1.0):
        """
        1) Calculates the local hinge loss
        2) If loss > 0, updates M by Eq. (6) in the paper:
           M <- M - gamma*( (x-x_pos)(x-x_pos)^T - (x-x_neg)(x-x_neg)^T )
        3) Projects M onto PSD to keep M valid (Theorem 1).
        Returns the hinge-loss value for reference.
        """
        diff_pos = (x - x_pos)
        diff_neg = (x - x_neg)
        d_pos = diff_pos.T @ self.M @ diff_pos
        d_neg = diff_neg.T @ self.M @ diff_neg
        loss_val = max(0.0, 1.0 + d_pos - d_neg)

        if loss_val > 0.0:
            A = np.outer(diff_pos, diff_pos) - np.outer(diff_neg, diff_neg)
            self.M = self.M - self.gamma * A
            self.M = project_psd(self.M)

        return loss_val

    def decompose(self):
        """
        Decomposes M into L such that M = L^T L (Cholesky factor).
        If small numerical issues arise, add a tiny eps on diagonal. 
        This is the step "Then, M is decomposed as L^T L" from the paper's text.
        """
        try:
            L = cholesky(self.M, lower=True)
        except np.linalg.LinAlgError:
            eps = 1e-8
            L = cholesky(self.M + eps*np.eye(self.d), lower=True)
        return L

    def get_M(self):
        """Getter for M (unchanged as requested)."""
        return self.M

    def set_M(self, M_new):
        """Setter for M (unchanged as requested)."""
        self.M = M_new.copy()

##############################################################################
# MLOML (Multilayer Online Metric Learning) EXACT forward steps from the paper:
#   1) local triplet loss & update M
#   2) M = L^T L decomposition
#   3) transform x^(i-1) -> x^(i) by L
#   4) optional activation
##############################################################################
class MLOML:
    def __init__(self, d, n_layers=3, gamma=0.01, activation='relu', lambda_reg=1e-4):
        """
        d         : input dimensionality
        n_layers  : number of metric layers
        gamma     : learning rate for each MOMLLayer
        activation: which nonlinearity to apply between layers: 'relu'|'sigmoid'|'tanh'
        lambda_reg: optional regularization coefficient (unused in minimal code, placeholders for BP).
        """
        self.d = d
        self.n_layers = n_layers
        self.gamma = gamma
        self.lambda_reg = lambda_reg

        # Activation function
        if activation == 'relu':
            self.activation_fn = relu
        elif activation == 'sigmoid':
            self.activation_fn = sigmoid
        elif activation == 'tanh':
            self.activation_fn = tanh
        else:
            raise ValueError("Unknown activation")

        # Build the metric layers
        self.layers = [MOMLLayer(d, gamma) for _ in range(n_layers)]

    def _forward_exact(self, x0, xp0, xq0):
        """
        Implements the EXACT forward pass from the paper:
         - For each layer i in [1..n_layers], do:
             (a) local hinge loss with M_i, update M_i (if >0)
             (b) L_i = chol(M_i)
             (c) x^(i) = L_i x^(i-1)
             (d) x^(i)_pos = L_i x^(i-1)_pos
             (e) x^(i)_neg = L_i x^(i-1)_neg
             (f) if i < n_layers, pass them through activation (ReLU/Sigmoid/tanh)
        Returns the final triple (x^(n), x_pos^(n), x_neg^(n)).
        """
        # We can store intermediate outputs for potential backprop
        x_list, xp_list, xq_list = [], [], []

        xi, xpi, xqi = x0, xp0, xq0
        for i in range(self.n_layers):
            layer = self.layers[i]

            # (a) local hinge-loss and update M_i
            #     But the paper's "exact" step does local update each time; let's do so.
            layer.local_update(xi, xpi, xqi)  # eq. (4)-(6)

            # (b) decompose M_i = L_i^T L_i
            L_i = layer.decompose()

            # (c)-(d)-(e) transform x^(i-1), x_pos^(i-1), x_neg^(i-1)
            xi_new  = L_i @ xi
            xpi_new = L_i @ xpi
            xqi_new = L_i @ xqi

            # (f) optional activation if not the last layer:
            if i < self.n_layers - 1:
                xi_new  = self.activation_fn(xi_new)
                xpi_new = self.activation_fn(xpi_new)
                xqi_new = self.activation_fn(xqi_new)

            # update for next iteration
            xi, xpi, xqi = xi_new, xpi_new, xqi_new

            # store intermediate
            x_list.append(xi)
            xp_list.append(xpi)
            xq_list.append(xqi)

        return x_list, xp_list, xq_list

    def fit_triplet(self, x, x_pos, x_neg):
        """
        Takes a single triplet (x, x_pos, x_neg), does EXACT forward pass from the paper.
        """
        x_list, xp_list, xq_list = self._forward_exact(x, x_pos, x_neg)

    def fit(self, pairs, labels, n_scans=1):
        """
        Trains MLOML over a list of pairs + labels. As in the user’s original code and
        the paper’s multi-scan approach (Sec. IV).
        
        pairs : [(query, sample), (query, sample), ...]
        labels: [ +1 or -1, +1 or -1, ... ]
        We assume every 2 consecutive pairs belong to the same query and form a triplet:
         (query, +sample, -sample) or (query, -sample, +sample).
        n_scans: number of times we pass over the entire data.
        """
        if len(pairs) % 2 != 0:
            raise ValueError("Number of pairs must be even (2 => 1 triplet).")

        for _ in range(n_scans):
            num_triplets = len(pairs)//2
            for i in range(num_triplets):
                query1, sample1 = pairs[2*i]
                query2, sample2 = pairs[2*i + 1]
                label1 = labels[2*i]
                label2 = labels[2*i + 1]

                # check queries match
                if not np.allclose(query1, query2):
                    raise ValueError("Consecutive pairs must share query to form a triplet.")

                # figure out which is positive or negative
                if label1 == 1 and label2 == -1:
                    x  = np.array(query1)
                    xp = np.array(sample1)
                    xq = np.array(sample2)
                elif label1 == -1 and label2 == 1:
                    x  = np.array(query1)
                    xp = np.array(sample2)
                    xq = np.array(sample1)
                else:
                    raise ValueError("Each triplet must have a +1 and -1 label.")

                self.fit_triplet(x, xp, xq)

    def get_mahalanobis_matrix(self):
        """
        Computes the final composite M = L^T L, 
        where L = (L_n ... L_1) is the sequential product of all layers' cholesky factors.
        (Paper, Section III).
        """
        L_total = np.eye(self.d)
        for layer in self.layers:
            M_i = layer.get_M()
            try:
                L_i = cholesky(M_i, lower=True)
            except np.linalg.LinAlgError:
                eps = 1e-8
                L_i = cholesky(M_i + eps*np.eye(self.d), lower=True)
            L_total = L_i @ L_total
        return L_total.T @ L_total
        
    def set_initial_matrix(self, M_init, layer_idx=None):
        """
        Set the initial Mahalanobis matrix.
        If layer_idx is None, set the initial matrix only for the first layer.
        Otherwise, set for the specified layer (0-indexed).
        """
        if layer_idx is None:
            self.layers[0].set_M(M_init)
        else:
            if 0 <= layer_idx < self.n_layers:
                self.layers[layer_idx].set_M(M_init)
            else:
                raise ValueError("Invalid layer index.")