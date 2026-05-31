import numpy as np

class OASIS:
    def __init__(self, C=0.07, enforce_psd=True, verbose=False):
        """
        Initializes the OASIS algorithm for learning a Mahalanobis (similarity) matrix.
        
        This implementation follows the paper "Large Scale Online Learning of Image Similarity Through Ranking"
        by Chechik et al. (2010), with the modification that we enforce the learned matrix to be PSD.
        
        Args:
            C (float): Aggressiveness parameter (corresponds to the maximum step size in the update).
            enforce_psd (bool): Whether to enforce the learned matrix to be Positive Semi-Definite (PSD).
            verbose (bool): If True, prints debug information.
        """
        self.C = C  # Aggressiveness parameter (from the paper, denoted as C)
        self.enforce_psd = enforce_psd
        self.verbose = verbose
        self.M = None  # Mahalanobis (similarity) matrix (W in the paper)

    def _ensure_psd(self, M):
        """
        Projects the matrix M onto the cone of Positive Semi-Definite (PSD) matrices.
        
        In Section 6.2 of the paper, the authors describe a PSD projection to guarantee that the learned
        similarity defines a valid Mahalanobis distance. Here we compute the eigen-decomposition and 
        clamp negative eigenvalues to zero.
        
        Args:
            M (ndarray): The matrix to project.
            
        Returns:
            ndarray: The projected PSD matrix.
        """
        # Compute eigen-decomposition of M
        eigvals, eigvecs = np.linalg.eigh(M)
        # Clamp negative eigenvalues to 0
        eigvals_clamped = np.maximum(eigvals, 0)
        # Reconstruct the matrix using the clamped eigenvalues
        M_psd = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
        if self.verbose:
            if np.all(eigvals >= 0):
                print("Matrix is already PSD.")
            else:
                print("Matrix projected to PSD.")
        return M_psd

    def partial_fit(self, M, triplets):
        """
        Performs online updates on the Mahalanobis matrix using training triplets.
        
        Each triplet is a tuple (p, p_plus, p_minus), corresponding to:
            - p: the anchor (query) example,
            - p_plus: a positive example (more similar to p),
            - p_minus: a negative example (less similar to p).
        
        The update is derived from the Passive-Aggressive algorithm as described in the paper:
        
            1. Compute the loss:
                 loss = 1 - p^T * M * (p_plus - p_minus)
               (This is the hinge loss; if loss <= 0, the constraint is satisfied.)
        
            2. If loss > 0, compute the gradient update direction:
                 V = p * (p_plus - p_minus)^T
               and its squared Frobenius norm: ||V||^2.
        
            3. Compute the step size (tau_i) as:
                 tau_i = min(C, loss / ||V||^2)
               (This is exactly Equation (9) in the paper.)
        
            4. Update the matrix:
                 M <- M + tau_i * V
        
        Finally, if enforce_psd is True, we project the updated matrix onto the PSD cone.
        
        Args:
            M (ndarray): The current Mahalanobis (similarity) matrix. If None, it is initialized
                         to the identity matrix.
            triplets (list of tuples): List of training triplets (p, p_plus, p_minus),
                         where each element is a 1-D numpy array.
        
        Returns:
            ndarray: The updated Mahalanobis matrix.
        """
        # Initialize M if not provided: set it to the identity matrix.
        if M is None:
            self.M = np.eye(len(triplets[0][0]))
        else:
            self.M = M
        
        # Process each triplet in an online fashion.
        for idx, (p, p_plus, p_minus) in enumerate(triplets):
            # Ensure inputs are numpy arrays
            p = np.array(p)
            p_plus = np.array(p_plus)
            p_minus = np.array(p_minus)
            
            # Compute the difference between positive and negative examples: delta = p_plus - p_minus
            delta = p_plus - p_minus
            
            # Compute the hinge loss:
            # loss = 1 - p^T * M * (p_plus - p_minus)
            loss = 1 - np.dot(p, np.dot(self.M, delta))
            
            # Only update if the loss is positive (i.e., if the margin constraint is violated)
            if loss > 0:
                # Compute the gradient matrix V = p * (p_plus - p_minus)^T
                V = np.outer(p, delta)
                # Compute the squared Frobenius norm of V: ||V||^2
                norm_squared = np.linalg.norm(V, 'fro') ** 2
                # Compute the step size tau_i = min(C, loss / ||V||^2)
                tau_i = min(self.C, loss / norm_squared)
                if self.verbose:
                    print(f"Triplet {idx}: loss = {loss:.4f}, tau = {tau_i:.4f}")
                # Update the Mahalanobis matrix according to the PA update rule:
                # M = M + tau_i * V
                self.M = self.M + tau_i * V
            
            # Enforce PSD projection after each update if required.
            if self.enforce_psd:
                self.M = self._ensure_psd(self.M)
        
        return self.M

    def get_matrix(self):
        """
        Returns the learned Mahalanobis (similarity) matrix.
        
        This is the final model after training.
        
        Returns:
            ndarray: The current Mahalanobis matrix.
        """
        return self.M