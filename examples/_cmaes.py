"""A compact CMA-ES for the examples (numpy only).

The standard (mu/mu_w, lambda)-CMA-ES of Hansen, "The CMA Evolution Strategy: A Tutorial"
(2016): ask() a population, simulate it as one batch, tell() it the costs. Minimises.
"""
import numpy as np


class CMAES:
    """es = CMAES(x0, sigma0, lam, rng); x = es.ask(); es.tell(costs) — minimising."""

    def __init__(self, x0, sigma, lam, rng):
        n = len(x0); self.n, self.lam, self.rng = n, lam, rng
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); self.w = w / w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m, self.sigma = np.asarray(x0, float), float(sigma)
        self.C, self.pc, self.ps, self.gen = np.eye(n), np.zeros(n), np.zeros(n), 0

    def ask(self):
        d, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d, 1e-20))
        self.z = self.rng.standard_normal((self.lam, self.n))
        self.y = self.z * self.D @ self.B.T
        return self.m + self.sigma * self.y

    def tell(self, f):
        idx = np.argsort(f)[:len(self.w)]
        yw = self.w @ self.y[idx]
        self.m = self.m + self.sigma * yw
        c_inv_sqrt_yw = self.B @ ((self.B.T @ yw) / self.D)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * c_inv_sqrt_yw
        self.gen += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) < (1.4 + 2 / (self.n + 1)) * self.chin
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (self.w[:, None, None] * np.einsum("ki,kj->kij", self.y[idx], self.y[idx])).sum(0)
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc)
                  + (1 - hs) * self.cc * (2 - self.cc) * self.C) + self.cmu * rank_mu)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))
