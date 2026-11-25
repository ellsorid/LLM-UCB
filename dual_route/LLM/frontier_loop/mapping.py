"""Sequential Bayesian linear readout on a fixed SOM feature space."""
import numpy as np


class KalmanMap:
    def __init__(self, dimension, context, prior_variance=0.25):
        if dimension < 1 or prior_variance <= 0:
            raise ValueError('Invalid prior')
        self.context = str(context)
        self.prior_mean = np.zeros(dimension)
        self.prior_mean[0] = .5
        self.prior_variance = float(prior_variance)
        self.mean = self.prior_mean.copy()
        self.cov = np.eye(dimension) * prior_variance
        self.events, self.observed = [], set()

    def predict(self, features):
        h = np.atleast_2d(features).astype(float)
        if h.shape[1] != len(self.mean) or not np.isfinite(h).all():
            raise ValueError('Invalid features')
        variance = np.einsum('ij,ij->i', h @ self.cov, h)
        return h @ self.mean, np.maximum(variance, 0)

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
        # Exact conditional simulation; avoids a dense Cholesky in >1300 dimensions.
        draw = self.prior_mean + np.sqrt(self.prior_variance) * rng.normal(size=len(self.mean))
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
            prior_variance=self.prior_variance, context=self.context,
            observed=np.asarray(sorted(self.observed)),
            event_h=np.asarray([e['h'] for e in events]).reshape(-1,len(self.mean)),
            event_gain=np.asarray([e['gain'] for e in events]).reshape(-1,len(self.mean)),
            event_meta=json.dumps([{k:v for k,v in e.items() if k not in ['h','gain']} for e in events]))

    @classmethod
    def load(cls, path):
        import json
        with np.load(path, allow_pickle=False) as r:
            obj = cls(len(r['mean']), str(r['context']), float(r['prior_variance']))
            obj.mean, obj.cov, obj.prior_mean = r['mean'].copy(),r['cov'].copy(),r['prior_mean'].copy()
            obj.observed = set(r['observed'].tolist())
            obj.events = [dict(**m,h=h.copy(),gain=g.copy()) for m,h,g in zip(json.loads(str(r['event_meta'])),r['event_h'],r['event_gain'])]
        return obj
