# %%
from sketchtools import SketchingOperator
from decoder import decoder_glasso
from utils import criteria
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from sklearn.covariance import graphical_lasso
from utils import generate_sparse_pd_precision
import random
import matplotlib.colors as mcolors

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino",
    "font.serif": ["Palatino"],
})

seed = 123
np.random.seed(seed)
random.seed(seed)

# %%
dim = 64
dataset = 'erdos'
# path_data='./data/'
#true_precision = np.load(path_data+dataset+'_d'+str(dim)+'.npy')[0]
true_precision = generate_sparse_pd_precision(dim, type_=dataset, M=4)
# %%
n_sample = 5000  # number of samples
reg = 9e-3
true_cov = np.linalg.pinv(true_precision, hermitian=True)
X = np.random.multivariate_normal(
    mean=np.zeros(dim), cov=true_cov, size=n_sample)
emp_cov = (1.0 / n_sample) * X.T.dot(X)

cov_glasso, precision_glasso = graphical_lasso(
    emp_cov, alpha=reg, max_iter=300, mode='cd', verbose=False, tol=1e-12)
# %% Sketching phase
M = 1024  # dim for the sketching, here only 50% of d^2/2 which is the size of the emp cov matrix
sketching_op = SketchingOperator(d=dim, M=M, method='structured')
y = sketching_op.sketch_covariance(emp_cov)  # sketch covariance
# %%
max_iter = 3000
max_iter_glasso = 10
S0 = np.eye(dim, dim)
reg2 = 9e-3
step_size = 1e-1
precision_sketching, res_sketching = decoder_glasso(y, sketching_op,
                                                    reg=reg2,
                                                    step_size=step_size,
                                                    max_iter=max_iter,
                                                    S0=S0,
                                                    verbose=True,
                                                    max_iter_glasso=max_iter_glasso,
                                                    n_verbose=5,
                                                    tolog=True,
                                                    )
# %%

vmax = max([np.max(true_precision),
            np.max(precision_sketching),
            np.max(precision_glasso)])
vmin = min([np.min(true_precision),
            np.min(precision_sketching),
            np.max(precision_glasso)])
rr = 1
fs = 20
thresh = 1e-8
nnz = np.sum(np.abs(true_precision) >= thresh)
cmap = plt.cm.bwr
norm = mcolors.TwoSlopeNorm(vmin=vmin / rr, vmax=vmax, vcenter=0)
toplot = {'Sketching M = {}'.format(M): precision_sketching,
          'Glasso': precision_glasso
          }
fig, ax = plt.subplots(1, len(toplot) +
                       1, figsize=(15, 5), constrained_layout=True)
ax[0].imshow(true_precision, cmap=cmap, norm=norm)
ax[0].set_title(
    r'True precision $\mathbf{{\Theta}}$,  $\mathrm{{nnz}}$ = {0}'.format(nnz),
    fontsize=fs)
ax[0].set_xticks([])
ax[0].set_yticks([])
i = 1
for k, mat in toplot.items():
    RE = criteria(true_precision, mat, type_='RE')
    im1 = ax[i].imshow(mat, cmap=cmap, norm=norm)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(
        '{0}, \n relative error = {1:.4f}'.format(
            k, RE), fontsize=fs)
    i += 1
divider = make_axes_locatable(ax[i - 1])
cax = divider.append_axes("right", size="3%", pad=0.03)
cb = fig.colorbar(im1, cax=cax, orientation='vertical')
ticks_loc = cb.ax.get_yticks().tolist()
cb.ax.yaxis.set_major_locator(mticker.FixedLocator([-0.5, 0, 1.5]))
cb.ax.tick_params(labelsize=18)
# %%
