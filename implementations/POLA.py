import numpy as np

class POLA:
    def __init__(self, dim, b_init=1.0, gamma=0.0):
        """
        Initialize the POLA model (Online Pseudo-metric Learning).
        
        Parameters:
            dim    : Dimensionality of the feature space.
            b_init : Initial threshold b₁.
            gamma  : Relaxation parameter for the inseparable case.
        
        According to the paper:
        - The algorithm starts with A₁ as the zero matrix.
        - b₁ is set to the provided b_init.
        """
        self.dim = dim
        self.b = b_init
        self.gamma = gamma
        
        # A₁ = 0 (zero matrix initialization as in the paper)
        self.A = np.zeros((dim, dim), dtype=float)
        
        # Optional custom vectorizer function for processing raw inputs.
        self.vectorizer = None

    def set_vectorizer(self, func):
        """Set a custom function to map raw inputs to numpy arrays."""
        self.vectorizer = func

    def _vectorize(self, x):
        """Convert input x to a numpy vector using the vectorizer if set; otherwise, use np.asarray."""
        if self.vectorizer is not None:
            return np.asarray(self.vectorizer(x), dtype=float)
        return np.asarray(x, dtype=float)

    def fit(self, pairs, labels, epochs=1, verbose=False):
        """
        Fit the POLA model using the online two-projection update rule described in the paper.
        
        The update for each instance pair (x, x') with label y is as follows:
        
        1. Compute the squared pseudo-distance:
           d² = vᵀ A v, where v = x - x'.
        
        2. Calculate the hinge loss:
           loss = max{0, y * (d² - b) + 1}.
           If loss is zero, no update is performed.
        
        3. If loss > 0, compute the step size:
           α = loss / (||v||⁴ + 1 + gamma).
        
        4. First Projection (onto the constraint set Cₜ):
           Update a temporary pair:
             Â = A - y * α * (v vᵀ)
             b̂ = b + y * α
           This forces the updated pair (Â, b̂) to have zero loss on the current example.
        
        5. Second Projection (onto the admissible set Cₐ, ensuring A is PSD and b ≥ 1):
           - If y = +1 (similar pair):
             • Set b ← b̂.
             • Compute the minimal eigenvalue (λ_min) and its eigenvector (u) of Â.
             • If λ_min < 0, project onto the PSD cone by setting:
                 A = Â - λ_min * (u uᵀ)
             • Otherwise, set A = Â.
           - If y = -1 (dissimilar pair):
             • No PSD projection is needed (since Â is already PSD).
             • Set A = Â and update b = max(b̂, 1).
        
        Args:
            pairs  : List of tuples ((x, x_prime)) representing instance pairs.
            labels : List of corresponding labels (+1 for similar, -1 for dissimilar).
            epochs : Number of passes over the data.
            verbose: If True, prints details of each update.
        """
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
            for idx, ((x, x_prime), y) in enumerate(zip(pairs, labels)):
                # Convert raw inputs to vectors.
                x_vec = self._vectorize(x)
                x_prime_vec = self._vectorize(x_prime)
                
                # Compute difference vector v = x - x'.
                v = x_vec - x_prime_vec
                
                # Compute squared pseudo-distance: d² = vᵀ A v.
                d2 = v @ self.A @ v
                
                # Compute hinge loss: loss = max{0, y*(d² - b) + 1}.
                loss = max(0.0, y * (d2 - self.b) + 1.0)
                if loss == 0.0:
                    # No update is needed when the loss is zero.
                    continue

                # Compute the squared norm of v and its fourth power.
                norm_v_sq = np.dot(v, v)
                norm_v_four = norm_v_sq ** 2
                
                # Compute step size α as specified in the paper:
                # α = loss / (||v||⁴ + 1 + gamma)
                alpha = loss / (norm_v_four + 1.0 + self.gamma)
                
                # --- First Projection Step: Project onto the constraint set Cₜ ---
                # Update temporary matrix and threshold:
                # Â = A - y * α * (v vᵀ)
                # b̂ = b + y * α
                A_hat = self.A - (y * alpha) * np.outer(v, v)
                b_hat = self.b + y * alpha
                
                # --- Second Projection Step: Project onto the admissible set Cₐ ---
                # Cₐ: Set of (A, b) where A is PSD and b ≥ 1.
                if y == 1:
                    # For similar pairs:
                    # 1. Update b to b̂.
                    self.b = b_hat
                    # 2. Project Â onto the PSD cone.
                    #    Compute minimal eigenvalue and its corresponding eigenvector.
                    eigvals, eigvecs = np.linalg.eigh(A_hat)
                    lambda_min = eigvals[0]   # np.linalg.eigh returns eigenvalues in ascending order.
                    u = eigvecs[:, 0]         # Corresponding eigenvector.
                    if lambda_min < 0:
                        # If the smallest eigenvalue is negative, fix Â:
                        # A = Â - λ_min * (u uᵀ)
                        self.A = A_hat - lambda_min * np.outer(u, u)
                    else:
                        # Otherwise, A is already PSD.
                        self.A = A_hat
                elif y == -1:
                    # For dissimilar pairs:
                    # Â is already PSD since the update is A + α*(v vᵀ),
                    # so we set A = Â and ensure b is at least 1.
                    self.A = A_hat
                    self.b = max(b_hat, 1.0)
                else:
                    raise ValueError("Label must be +1 or -1")
                
                if verbose:
                    print(f"Update {idx}: loss={loss:.4f}, alpha={alpha:.4f}, y={y}")
        if verbose:
            print("Training complete.")

    def predict(self, x, x_prime):
        """
        Predict the similarity for a pair (x, x_prime).
        
        According to the paper, the prediction is:
            +1 if (d_A(x, x_prime))² ≤ b,
            -1 otherwise.
        """
        x_vec = self._vectorize(x)
        x_prime_vec = self._vectorize(x_prime)
        v = x_vec - x_prime_vec
        d2 = v @ self.A @ v
        return 1 if d2 <= self.b else -1

    def get_mahalanobis_matrix(self):
        """Return the current Mahalanobis (PSD) matrix A."""
        return self.A

    def set_mahalanobis_matrix(self, matrix):
        """Manually override the matrix A."""
        self.A = matrix