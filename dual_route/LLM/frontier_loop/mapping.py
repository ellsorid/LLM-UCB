"""Sequential Bayesian linear readout on a fixed SOM feature space."""
import numpy as np


def som_laplacian_prior(topology, intercept_var=0.25, node_var=0.15, laplacian_weight=1.5):
    """Build a block-diagonal SPD prior covariance using SOM graph-Laplacian smoothing."""
    if hasattr(topology, "detach"):
        topo = topology.detach().cpu().numpy().astype(float)
    else:
        topo = np.asarray(topology, dtype=float)
    if topo.ndim != 2 or topo.shape[0] != topo.shape[1] or topo.shape[0] < 1 or not np.isfinite(topo).all():
        raise ValueError("Topology must be a finite square matrix")
    if intercept_var <= 0 or node_var <= 0 or laplacian_weight < 0:
        raise ValueError("Prior variances and Laplacian weight must be positive/nonnegative")
    k = topo.shape[0]
    adj = ((topo > 0) & (topo <= 1.0 + 1e-9)).astype(float)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    laplacian = np.diag(adj.sum(axis=1)) - adj
    precision = np.eye(k, dtype=float) + float(laplacian_weight) * laplacian
    cov_nodes = np.linalg.inv(precision)
    cov_nodes = 0.5 * (cov_nodes + cov_nodes.T)
    mean_diag = max(float(np.mean(np.diag(cov_nodes))), 1e-12)
    cov_nodes *= float(node_var) / mean_diag
    prior_cov = np.zeros((k + 1, k + 1), dtype=float)
    prior_cov[0, 0] = float(intercept_var)
    prior_cov[1:, 1:] = cov_nodes
    return prior_cov


def som_manifold_features(z, centers, bandwidth):
    """Project an embedding vector onto the frozen SOM simplex basis [1, simplex]."""
    z_arr = np.asarray(z, dtype=float).ravel()
    c_arr = np.asarray(centers, dtype=float)
    if c_arr.ndim != 2 or c_arr.shape[1] != len(z_arr):
        raise ValueError("Incompatible SOM center and feature dimensions")
    bw = max(float(bandwidth), 1e-6)
    d2 = ((c_arr - z_arr) ** 2).sum(axis=1)
    cell = int(d2.argmin())
    weights = np.exp(-d2 / (2.0 * bw))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 1e-12:
        weights = np.exp(-(d2 - float(d2.min())) / (2.0 * bw))
        total = float(weights.sum())
    simplex = weights / max(total, 1e-12)
    psi = np.r_[1.0, simplex]
    return cell, simplex, psi


class KalmanMap:
    def __init__(self, dimension, context, prior_variance=0.25, prior_cov=None):
        if dimension < 1 or prior_variance <= 0:
            raise ValueError('Invalid prior')
        self.context = str(context)
        self.prior_mean = np.zeros(dimension, dtype=float)
        self.prior_mean[0] = .5
        self.prior_variance = float(prior_variance)
        if prior_cov is None:
            self.prior_cov = np.eye(dimension, dtype=float) * self.prior_variance
        else:
            pc = np.asarray(prior_cov, dtype=float)
            if pc.shape != (dimension, dimension) or not np.isfinite(pc).all():
                raise ValueError('Invalid prior covariance shape or values')
            if not np.allclose(pc, pc.T, rtol=1e-6, atol=1e-8):
                raise ValueError('Prior covariance must be symmetric')
            pc = 0.5 * (pc + pc.T)
            try:
                np.linalg.cholesky(pc)
            except np.linalg.LinAlgError as exc:
                raise ValueError('Prior covariance must be positive definite') from exc
            self.prior_cov = pc
        self.mean = self.prior_mean.copy()
        self.cov = self.prior_cov.copy()
        self.events, self.observed = [], set()

    def predict(self, features):
        h = np.atleast_2d(features).astype(float)
        if h.shape[1] != len(self.mean) or not np.isfinite(h).all():
            raise ValueError('Invalid features')
        variance = np.einsum('ij,ij->i', h @ self.cov, h)
        return h @ self.mean, np.maximum(variance, 0)

    def cell_surface(self):
        """Evaluate posterior mean and standard deviation at each SOM prototype basis vector."""
        k = len(self.mean) - 1
        if k < 1:
            raise ValueError('SOM manifold dimension requires at least 1 cell')
        basis = np.zeros((k, len(self.mean)), dtype=float)
        basis[:, 0] = 1.0
        basis[np.arange(k), 1 + np.arange(k)] = 1.0
        means, variances = self.predict(basis)
        return means, np.sqrt(np.maximum(variances, 0.0))

    def update(self, uid, features, score, noise_variance, context, process_variance=0):
        h = np.asarray(features, dtype=float)
        if context != self.context:
            raise ValueError('Rebuild/reconcile mapping before changing score or representation context')
        if uid in self.observed:
            raise ValueError('Duplicate label; intentional repeat requires a distinct observation ID')
        if h.shape != self.mean.shape or not np.isfinite(h).all():
            raise ValueError('Invalid observation features')
        if not np.isfinite([score, noise_variance, process_variance]).all() or not 0 <= score <= 1 or noise_variance <= 0 or process_variance < 0:
            raise ValueError('Invalid normalized score/noise')
        if process_variance:
            self.cov.flat[::len(self.mean)+1] += process_variance
        v = self.cov @ h
        innovation_variance = float(h @ v + noise_variance)
        gain = v / innovation_variance
        residual = float(score - h @ self.mean)
        self.mean += gain * residual
        self.cov -= np.outer(v, v) / innovation_variance
        self.cov = (self.cov + self.cov.T) * .5
        self.events.append(dict(h=h.copy(), gain=gain.copy(), score=float(score), noise=float(noise_variance), q=float(process_variance)))
        self.observed.add(uid)
        return dict(residual=residual, innovation_variance=innovation_variance, gain_norm=float(np.linalg.norm(gain)))

    def sample_weights(self, rng):
        # Exact conditional simulation with correlated prior covariance.
        chol = np.linalg.cholesky(self.prior_cov + 1e-12 * np.eye(len(self.mean)))
        draw = self.prior_mean + chol @ rng.normal(size=len(self.mean))
        for event in self.events:
            if event['q']:
                draw += np.sqrt(event['q']) * rng.normal(size=len(draw))
            simulated = event['h'] @ draw + np.sqrt(event['noise']) * rng.normal()
            draw += event['gain'] * (event['score'] - simulated)
        return draw

    def delta(self, parent, child):
        return tuple(float(v[0]) for v in self.predict(np.asarray(child)-np.asarray(parent)))

    def save(self, path):
        import json
        events = self.events
        np.savez_compressed(path, mean=self.mean, cov=self.cov, prior_mean=self.prior_mean,
            prior_variance=self.prior_variance, prior_cov=self.prior_cov, context=self.context,
            observed=np.asarray(sorted(self.observed)),
            event_h=np.asarray([e['h'] for e in events]).reshape(-1,len(self.mean)),
            event_gain=np.asarray([e['gain'] for e in events]).reshape(-1,len(self.mean)),
            event_meta=json.dumps([{k:v for k,v in e.items() if k not in ['h','gain']} for e in events]))

    @classmethod
    def load(cls, path):
        import json
        with np.load(path, allow_pickle=False) as r:
            prior_cov = r['prior_cov'].copy() if 'prior_cov' in r.files else None
            obj = cls(len(r['mean']), str(r['context']), float(r['prior_variance']), prior_cov=prior_cov)
            obj.mean, obj.cov, obj.prior_mean = r['mean'].copy(),r['cov'].copy(),r['prior_mean'].copy()
            obj.observed = set(r['observed'].tolist())
            obj.events = [dict(**m,h=h.copy(),gain=g.copy()) for m,h,g in zip(json.loads(str(r['event_meta'])),r['event_h'],r['event_gain'])]
        return obj
