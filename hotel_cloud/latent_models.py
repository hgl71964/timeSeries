import numpy as np
from copy import deepcopy
from numba import jit 

class discrete_latent_markov:

    """
    discrete state space latent markov model to cluster time series
        parameter learning via EM 
    """

    def __init__(self,
                data: np.ndarray,  # shape (n, T); n: number of sequence, T: length of sequence 
                n_cluster: int,   # number of cluster
                bound: tuple,  #  (lower, upper) -> lower and upper bound for discrete space
                init_strategy: str = None,  # specify prior distribution
                ):
    
        self.lower, self.upper = bound[0], bound[1]
        self.n, self.seq_len = data.shape[0], data.shape[1]
        self.n_cluster = n_cluster
        
        if init_strategy is None:
            pz = np.random.rand(n_cluster)  # random prior
        else:
            pz = np.array([1/n_cluster] * n_cluster, dtype=np.float64)  # uniform prior
            
        pz /= np.sum(pz)

        state_space = int(self.upper - self.lower + 1)
        pv1gz = np.random.rand(state_space, n_cluster); pv1gz = np.divide(pv1gz, pv1gz.sum(axis=0)); 
        pvgvz = np.random.rand(state_space, state_space, n_cluster); pvgvz = np.divide(pvgvz, pvgvz.sum(axis=0)); 

        self.pz = pz; self.pv1gz = pv1gz; self.pvgvz = pvgvz
        
        # check probability distribution 
        assert np.isclose(self.pz.sum(), 1.)
        assert np.isclose(self.pv1gz.sum(axis=0).all(), 1.)
        assert np.isclose(self.pvgvz[:,:,0].sum(axis=0).all(), 1.) # pvgvz[:,:,0] is the first transition matrix

    @jit(nopython=True)
    def e_step(self, data, qz):
        for i in range(self.n):  
            """
            pz -> prior latent distribution; shape (n_cluster, )
            qz -> posterior latent distribution for each sequence; shape(n, n_cluster)

            updates:
            q_i(z) = p(z|x_i; \theta) \propto p(z, x_i; \theta)/p(x_i; \theta);
            notice the \propto means we need to re-normalised the probability
            """
            # operation in log-space to prevent overflow
            index = int(data[i,0] - self.lower)
            log_qz = np.log(self.pz) + np.log(self.pv1gz[index])

            for t in range(1, self.seq_len):
                idx1, idx2 = int(data[i][t-1]- self.lower), int(data[i][t] - self.lower)

                log_qz += np.log(self.pvgvz[idx2, idx1])

            qz[i] = np.exp(log_qz)
        return qz

    @jit(nopython=True) 
    def log_llike(self, qz):
        llike = np.sum(np.log(np.sum(qz, axis=1))) 
        return llike

    @jit(nopython=True)
    def normalise(self, qz):
        normalise_qz = qz/qz.sum(axis=1).reshape(-1,1)
        assert (np.isclose(normalise_qz.sum(axis=1).all(), 1.))
        return normalise_qz

    @jit(nopython=True)
    def m_step(self, data, qz):
        
        state_space = int(self.upper - self.lower + 1)

        # update pz -> argmax KL-divergence
        pz = qz.sum(axis=0); pz/= np.sum(pz)

        # update pv1gz
        pv1gz = np.empty((state_space, self.n_cluster), dtype = np.float64)
        
        for i in range(state_space):
            index = (data[:,0] == int(i + self.lower))
            pv1gz[i] = np.sum(qz[index], axis=0)
        
        pv1gz = np.divide(pv1gz, pv1gz.sum(axis=0))  # normalisation

        # update pvgvz
        pvgvz = np.zeros((state_space, state_space, self.n_cluster), dtype = np.float64)

        for i in range(state_space):
            for j in range(state_space):
                for k in range(self.n):
                    count=0
                    for t in range(1, self.seq_len):
                        if (int(data[k][t] - self.lower) == i) and (int(data[k][t-1] - self.lower) == j):
                            count+=1
                    pvgvz[i, j, :] += qz[k] * count
                
        pvgvz = np.divide(pvgvz, pvgvz.sum(axis=0))

        self.pz = pz; self.pv1gz = pv1gz; self.pvgvz = pvgvz
        
    @jit(nopython=True)
    def run_epoch(self,
                data: np.ndarray,  #  shape (n, seq_len); n: number of sequence, seq_len: length of sequence 
                epochs: int,
                verbose: bool = False, 
                ):

        llike = 0  # start log-likelihood
        qz = np.empty((self.n, self.n_cluster), dtype=np.float64)  # variational distribution for each sequence 

        for e in range(epochs):
            qz = self.e_step(data, qz)
            new_llike = self.log_llike(qz)

            if verbose:
                print(f"Iter {e+1}, log-likelihood {new_llike:,.2f}")

            llike = new_llike
            qz = self.normalise(qz)
            posterior_z = deepcopy(qz)

            self.m_step(data, qz)
        
        return posterior_z, llike 