#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import numpy as np
from analytic import compute_analytic
import os
import pandas as pd
import scienceplots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
plt.style.use('science')

Adam_epoch = 20001
LBFGS_epoch = 1001
lr_A = 1e-4
lr_L = 1.0

# Thông số ống
X_outlet = 2.25
X_throat = 1.5

value_pb = 0.85 #THING TO DEFINE

w_f1 = 4.0   # Continuity (Bảo toàn khối lượng)
w_f2 = 60.0  # Momentum (Bảo toàn động lượng)
w_f3 = 4.0  # Energy (Bảo toàn năng lượng)
w_f4 = 4.0   # Ideal gas

w_pde = 1.0
w_inlet = 0.0       # 10.0 - 50.0
w_throat = 1.0
w_outlet = 0.0      # 10.0 - 100.0

# Choked
value_p_throat = 0.528
value_rho_throat = 0.634
value_T_throat = 0.833
value_v_throat = 1.0

# Non-choked
# value_p_throat = 0.84
# value_rho_throat = 0.88
# value_T_throat = 0.95   
# value_v_throat = 0.54

n