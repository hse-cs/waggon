## Description

An intuitive approach for uncertainty quantifi- cation is using an ensemble surrogate model, e.g., an adversarial deep ensemble (DE) [4]. Single predictors of the ensemble are expected to agree on their predictions over observed regions of the fea- ture space, i.e., where data are given and so the uncertainty is low and vice versa. The further these single predictors get from known regions of the feature space, the greater the discrepancy in their predictions.

## Usage

```python
from waggon.surrogates import DE
```

[4] Lakshminarayanan, B., Pritzel, A. and Deepmind, C. (n.d.). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. PMLR. Available at: https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf.
‌