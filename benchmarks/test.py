import conv_nn as conv

from waggon import functions as f
from waggon.optim import SurrogateOptimiser
from waggon.surrogates import GP, DGP
from waggon.acquisitions import CB


opt = SurrogateOptimiser(
    func=conv.ConvNN(),
    surr=GP(n_epochs=10),
    acqf=CB(),
    seed=2,
    max_iter=10,
    plot_results=True,
    verbose=2
)

result = opt.optimise()
print(result)