
from sklearn.cross_decomposition import CCA
import numpy as np
import pickle
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

import sys
import kinematics_preprocessing_specs

import config


#Check which configs we need to add
parser = config.config_parser()
args, unknown = parser.parse_known_args()

#number of timesteps?
n_fixedsteps = 100

#Load unprocessed activation data from neural network
##Training and testing or just testing?
data = np.load(args.testing_save_path, allow_pickle=True)
activity = data.item()

#conditions would be left and right reaches?

#number of timesteps?
n_fixedsteps = 100

pca_out = PCA(n_components = 3)

#plot

colors = plt.cm.ocean(np.linspace(0,1,8))
ax = plt.figure(dpi = 100).add_subplot(projection='3d')

ax.plot(pca_out)

ax.grid(False)
plt.grid(b = None)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')

plt.show()


