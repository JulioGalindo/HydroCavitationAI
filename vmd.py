
"""Variational Mode Decomposition (VMD) implementation.

Reference:
Dragomiretskiy & Zosso, IEEE Trans. Signal Process., 2014.
Optimisation of hyper‑parameters is performed externally via SSA.
"""
import numpy as np

class VMD:
    """Compute VMD modes of an input signal."""

    def __init__(self, alpha: float = 2000, tau: float = 0.0, K: int = 6,
                 DC: bool = False, init: str = 'uniform', tol: float = 1e-7, N_iter: int = 500):
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol
        self.N_iter = N_iter

    def _mirror(self, f):
        return np.concatenate([np.flip(f), f, np.flip(f)])

    def decompose(self, signal: np.ndarray):
        """Perform VMD.

        Returns
        -------
        modes : np.ndarray
            Array of shape (K, N) with decomposed modes.
        """
        fs = len(signal)
        f_hat = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(fs, d=1/fs))

        # Initialise
        if self.init == 'uniform':
            omega = np.linspace(0, 0.5, self.K, endpoint=False)
        else:
            omega = np.random.rand(self.K)

        u_hat = np.zeros((self.K, fs), dtype=complex)
        lambda_hat = np.zeros(fs, dtype=complex)

        # Iterations
        for _ in range(self.N_iter):
            u_hat_prev = u_hat.copy()

            for k in range(self.K):
                residual = f_hat - u_hat.sum(axis=0) + u_hat[k]
                u_hat[k] = residual * (1 + self.alpha * (freq - omega[k])**2) ** -1

                if not self.DC or k != 0:
                    omega[k] = np.sum(freq * np.abs(u_hat[k])**2) / (np.sum(np.abs(u_hat[k])**2) + 1e-12)

            lambda_hat = lambda_hat + self.tau * (f_hat - u_hat.sum(axis=0))

            # Check convergence
            if np.sum(np.abs(u_hat - u_hat_prev)**2) / np.sum(np.abs(u_hat_prev)**2 + 1e-12) < self.tol:
                break

        modes = np.real(np.fft.ifft(np.fft.ifftshift(u_hat, axes=1), axis=1))
        return modes
