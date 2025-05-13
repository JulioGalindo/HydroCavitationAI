# Auto-generated M4‑optimised module – keep lines tight

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

class VMD:
    def __init__(self, alpha=2000, K=6, tau=0, dc=False, tol=1e-7, N_iter=500):
        self.alpha, self.K, self.tau, self.dc, self.tol, self.N_iter = alpha,K,tau,dc,tol,N_iter
    def decompose(self,x):
        x=x.astype(float); N=len(x)
        f_hat=fftshift(fft(x)); f_plus=f_hat.copy(); f_plus[:N//2]=0
        u_hat=np.zeros((self.K,N),complex); omega=np.arange(self.K)*0.5/self.K
        lam=np.zeros(N); freqs=fftfreq(N); diff=self.tol+1; n=0
        while diff>self.tol and n<self.N_iter:
            u_old=u_hat.copy()
            for k in range(self.K):
                sum_=f_plus-u_hat.sum(0)+u_hat[k]
                denom=1+self.alpha*(freqs-omega[k])**2 if (self.dc==False or k) else 1+self.alpha*freqs**2
                u_hat[k]=(sum_+lam/2)/denom
                if self.dc==False or k:
                    omega[k]=np.sum(np.abs(u_hat[k])**2*freqs)/(np.sum(np.abs(u_hat[k])**2)+1e-12)
            lam+=self.tau*(f_plus-u_hat.sum(0))
            diff=np.linalg.norm(u_hat-u_old)/(np.linalg.norm(u_old)+1e-12); n+=1
        return np.real(ifft(ifftshift(u_hat,1),1))
