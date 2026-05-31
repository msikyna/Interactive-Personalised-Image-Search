import numpy as np

def soft_threshold(X, thresh):
    """
    Elementwise soft-thresholding operator for ℓ1 regularization.
    
    For each entry X[i,j], this does:
        X[i,j] <- sign(X[i,j]) * max(|X[i,j]| - thresh, 0).
    """
    return np.sign(X) * np.maximum(np.abs(X) - thresh, 0)

def soft_threshold_offdiag(X, thresh):
    """
    Soft-thresholding for off-diagonal elements only.
    Diagonal entries are left unchanged.

    This corresponds to ℓ1 regularization on off-diagonal
    elements (the 'offdiag' version in the paper).
    """
    Y = np.copy(X)
    d = X.shape[0]
    for i in range(d):
        for j in range(d):
            if i != j:
                val = X[i, j]
                # Apply soft threshold
                Y[i, j] = np.sign(val) * max(abs(val) - thresh, 0)
    return Y


class SORS:
    """
    Sparse Online Relative Similarity (SORS)

    Reimplementation of the first-order (proximal gradient) approach
    exactly as in the paper, but with triplet inputs: (x, x+, x-).

    Parameters:
        eta       : Learning rate (step size).
        reg_lambda: Regularization parameter λ for controlling sparsity.
        reg_type  : Either 'l1' for standard ℓ1 on all entries of M,
                    or 'offdiag' to apply ℓ1 only on off-diagonal elements.
        M_init    : Initial Mahalanobis matrix. If None, identity will be used
                    the first time we see a sample dimension.
    """
    def __init__(self, eta=0.1, reg_lambda=1e-3, reg_type='l1', M_init=None):
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type  # 'l1' or 'offdiag'
        self.M = M_init           # Mahalanobis matrix

    def fit(self, triplets, epochs=1):
        """
        Fit the SORS model using triplets of the form:
            (x, x+, x-)
        with the hinge loss: [1 - (x^T M x+  -  x^T M x-)]_+.

        Update steps follow the paper (SORS = first-order):
          1) Compute gradient G if margin is violated.
          2) M_temp = M - eta * G
          3) Apply soft-thresholding operator (prox) for sparsity.

        Args:
            triplets : iterable of (x, x_plus, x_minus),
                       each x in R^d as a numpy array
            epochs   : number of passes over the data

        Returns:
            self (the fitted model)
        """
        for epoch in range(epochs):
            for (x, x_pos, x_neg) in triplets:
                # Ensure data is numpy array
                x = np.asarray(x)
                x_pos = np.asarray(x_pos)
                x_neg = np.asarray(x_neg)

                # If M is None at first usage, initialize to identity
                if self.M is None:
                    d = x.shape[0]
                    self.M = np.eye(d)

                # Compute the hinge loss margin:
                # margin = 1 - ( x^T M x+  -  x^T M x- )
                score_pos = x.T.dot(self.M).dot(x_pos)
                score_neg = x.T.dot(self.M).dot(x_neg)
                margin = 1.0 - score_pos + score_neg

                # If margin > 0, the gradient is:
                #   ∂/∂M [−(x^T M x+ - x^T M x−)] = x x−^T - x x+^T
                # If margin <= 0, gradient is 0
                if margin > 0:
                    G = np.outer(x, x_neg) - np.outer(x, x_pos)
                else:
                    G = np.zeros_like(self.M)

                # 1) Gradient step
                M_temp = self.M - self.eta * G

                # 2) Proximal step (soft thresholding) for sparsity
                thresh = self.eta * self.reg_lambda
                if self.reg_type == 'l1':
                    self.M = soft_threshold(M_temp, thresh)
                elif self.reg_type == 'offdiag':
                    self.M = soft_threshold_offdiag(M_temp, thresh)
                else:
                    # If reg_type is unknown, no thresholding
                    self.M = M_temp

        return self

    def get_mahalanobis_matrix(self):
        """
        Return the current Mahalanobis (similarity) matrix M.
        """
        return self.M


class AdaSORS:
    """
    Adaptive Sparse Online Relative Similarity (AdaSORS)

    Reimplementation of the second-order approach from the paper,
    using triplets (x, x+, x-) and an AdaGrad-like update:
        1) If margin is violated, G = x x-^T - x x+^T
        2) H = sqrt(H^2 + G^2) (element-wise)
        3) M_temp = M - (eta * G) / (delta + H)
        4) M_new = prox( M_temp ), applying element-wise or off-diag threshold

    Parameters:
        eta        : Base learning rate.
        reg_lambda : Regularization parameter λ for controlling sparsity.
        delta      : Small constant added to keep each element of (delta + H) invertible.
        reg_type   : 'l1' for full ℓ1, or 'offdiag' for off-diagonal only.
        M_init     : Initial matrix M (if None, we initialize to identity).
    """
    def __init__(self, eta=0.1, reg_lambda=1e-3, delta=1e-6, 
                 reg_type='l1', M_init=None):
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.delta = delta
        self.reg_type = reg_type
        self.M = M_init
        self.H = None  # second-order accumulator

    def fit(self, triplets, epochs=1):
        """
        Fit the AdaSORS model with triplets:
            (x, x+, x-)

        Hinge loss: [1 - ( x^T M x+ - x^T M x- )]_+.
        If margin > 0, gradient G = x x-^T - x x+^T. Then:
          H = sqrt(H^2 + G^2),   Sigma = delta + H,
          M_temp = M - (eta * G) / Sigma,
          M_new  = prox(M_temp).

        Args:
            triplets : iterable of (x, x_plus, x_minus)
            epochs   : number of passes over the data

        Returns:
            self
        """
        for epoch in range(epochs):
            for (x, x_pos, x_neg) in triplets:
                x = np.asarray(x)
                x_pos = np.asarray(x_pos)
                x_neg = np.asarray(x_neg)

                # Initialize M (and H) if necessary
                if self.M is None:
                    d = x.shape[0]
                    self.M = np.eye(d)
                if self.H is None:
                    d = self.M.shape[0]
                    self.H = np.zeros((d, d))

                # Compute margin
                score_pos = x.T.dot(self.M).dot(x_pos)
                score_neg = x.T.dot(self.M).dot(x_neg)
                margin = 1.0 - score_pos + score_neg

                # Compute gradient if margin is violated
                if margin > 0:
                    G = np.outer(x, x_neg) - np.outer(x, x_pos)
                else:
                    G = np.zeros_like(self.M)

                # Update H (elementwise)
                #   H[i,j] = sqrt( H[i,j]^2 + G[i,j]^2 )
                self.H = np.sqrt(self.H**2 + G**2)

                # Sigma = delta + H, also elementwise
                Sigma = self.delta + self.H

                # M_temp = M - (eta * G) / Sigma
                # Divide G elementwise by Sigma
                M_temp = self.M - (self.eta * G / Sigma)

                # Next, apply the proximal operator
                #   soft-threshold each element with threshold = (eta * reg_lambda) / Sigma
                if self.reg_type == 'l1':
                    # Full ℓ1 threshold
                    # proximal step is sign(M_temp)*max(|M_temp|- threshold, 0)
                    # but threshold here is elementwise: (eta*reg_lambda)/Sigma
                    T = (self.eta * self.reg_lambda) / Sigma
                    # sign(M_temp)*max(abs(M_temp)-T, 0)
                    M_new = np.sign(M_temp) * np.maximum(np.abs(M_temp) - T, 0)
                    self.M = M_new

                elif self.reg_type == 'offdiag':
                    # Off-diagonal threshold only
                    M_new = np.copy(M_temp)
                    d = M_temp.shape[0]
                    for i in range(d):
                        for j in range(d):
                            if i != j:
                                # threshold is (eta*reg_lambda)/Sigma[i,j]
                                thresh = (self.eta * self.reg_lambda) / Sigma[i,j]
                                val = M_temp[i,j]
                                M_new[i,j] = np.sign(val) * max(abs(val) - thresh, 0)
                    self.M = M_new
                else:
                    # If reg_type unknown, skip thresholding
                    self.M = M_temp

        return self

    def get_mahalanobis_matrix(self):
        """
        Return the current Mahalanobis matrix M.
        """
        return self.M