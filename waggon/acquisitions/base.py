class Acquisition:
    def __init__(self):
        '''
        Base class for an acquisition function.

        Parameters
        ----------
        y : np.array of shape (n_samples * func.n_obs, 1)
            Target values of the black-box function.
        
        name : str
            Name of the acquisition function.
        
        surr : # TODO: add type
            Surrogate model to compute the acquisition function.
        '''
        super(Acquisition, self).__init__()

        self.y     = None
        self.name  = None
        self.surr  = None
        self.conds = None
    
    def __call__(self):
        '''
        Call of the acquisition function.

        Parameters
        ----------
        x : np.array of shape (n_samples * func.n_obs, func.dim)
            Candidate points.

        Returns
        -------
        Acquisition function value.
        '''
        pass