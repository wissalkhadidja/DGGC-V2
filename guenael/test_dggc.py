#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:05:57 2026

@author: guenael
"""

import torch
import numpy as np
from dggc_exp import Experience, GENERATORS
from dggc_exp.models import fit_gaussian_reference
import matplotlib.pyplot as plt


### Config

N = 5000

### Data generation

X, y = GENERATORS["gaussian_blobs"](N, 2, 635465)

### Class initialisation
test = Experience(X, y)

test.visu_embeddings('test', embedding="initial", color_by="classes")

### NCE training
test.train_NCE()

### NCE test

x = torch.tensor(test.X0, dtype=torch.float32, device=test.config.device)
x = x.detach().requires_grad_(True)
den_ratio = test.density_model(x).detach().numpy() # ratio

mu, cov = fit_gaussian_reference(test.X0)

from scipy.stats import multivariate_normal
var = multivariate_normal(mean=mu, cov=cov)
den_ref = var.pdf(test.X0).reshape([N,1])


den_X = den_ref * np.exp(den_ratio)

plt.figure()
plt.scatter(test.X0[:,0], test.X0[:,1], c=y)

plt.figure()
plt.scatter(test.X0[:,0], test.X0[:,1], c=np.exp(den_ratio))

plt.figure()
plt.scatter(test.X0[:,0], test.X0[:,1], c=den_ref)

plt.figure()
plt.scatter(test.X0[:,0], test.X0[:,1], c=den_X)


X_ref = test._make_reference(test.X0)
plt.figure()
plt.scatter(X_ref[:,0], X_ref[:,1], c=var.pdf(X_ref).reshape([N,1]))

plt.figure()
plt.scatter(test.X0[:,0], test.X0[:,1], c=y)
plt.scatter(X_ref[:,0], X_ref[:,1], alpha=0.2)



