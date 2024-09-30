from sklearn.base import BaseEstimator

class Surrogate(BaseEstimator):
    '''
    Surrogate base class.
    '''
    def __init__(self):
        super(Surrogate, self).__init__()
    
    def fit(self, X, y):
        '''
        Fit the surrogate model on a training set (X, y).

        Parameters
        ----------
        X : np.array of shape (n_samples * func.n_obs, func.dim)
            Training input points.

        y : np.array of shape (n_samples * func.n_obs, func.dim)
            Target values of the black-box function.
        '''
        pass
    
    def predict(self, X):
        '''
        Preedict black-box function values.

        Parameters
        ----------
        X : np.array of shape (n_samples * func.n_obs, func.dim)
            Candidate points.
        
        Returns
        -------
        y : np.array of shape (n_samples * func.n_obs, 1)
            Predicted values of the black-box function
        '''
        pass

class GenSurrogate(Surrogate):
    '''
    Generative surrogate base class.
    '''
    def __init__(self):
        super(GenSurrogate, self).__init__()
        pass
    
    def sample(self, X):
        '''
        Generate black-box response.

        Parameters
        ----------
        X : np.array of shape (n_samples * func.n_obs, func.dim)
            Candidate points.
        
        Returns
        -------
        y : np.array of shape (n_sample * func.n_obs, 1)
            Generated response.
        '''
        pass
