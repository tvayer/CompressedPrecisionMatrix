# %%
import numpy as np
import time
import matplotlib.pyplot as plt
# %%


def return_blocks(p, col):
    indices = np.arange(p)
    indices_minus_col = np.concatenate(
        [indices[:col], indices[col + 1:]])
    _11 = indices_minus_col[:, None], indices_minus_col[None]
    _12 = indices_minus_col, col
    _21 = col, indices_minus_col
    _22 = col, col
    return _11, _12, _21, _22


def slope(x, y):
    if len(x) < 4 or len(y) < 4:
        raise ValueError("Il faut au moins 4 points pour calculer la pente.")
    x_dernier = x[-4:]
    y_dernier = y[-4:]
    slope, intercept = np.polyfit(x_dernier, y_dernier, 1)
    return slope, intercept


def _outer_prod(x, y):
    return np.outer(x, y)


# %%
all_p = [500, 1000, 2000, 3000, 5000, 10000]
repeat = 3

speed_mat_prod = np.zeros((repeat, len(all_p)))
speed_square = np.zeros((repeat, len(all_p)))
speed_square_and_insert = np.zeros((repeat, len(all_p)))
speed_square_and_insert_with_perturb = np.zeros((repeat, len(all_p)))
speed_square_and_insert_with_access = np.zeros((repeat, len(all_p)))

for i in range(repeat):
    for j, p in enumerate(all_p):
        A = np.random.randn(p, p)
        B = np.random.randn(p, p)
        v = np.random.randn(p - 1)
        w = np.random.randn(p - 1)
        _11, _12, _21, _22 = return_blocks(p, 0)

        # st = time.time()
        # _ = B @ A @ B
        # ed = time.time()
        # speed_mat_prod[i, j] = ed-st

        # A_11 = A[_11]
        # st = time.time()
        # _ = np.inner(v,  A_11 @ v)*_outer_prod(v, v)
        # ed = time.time()
        # speed_square[i, j] = ed-st

        st = time.time()
        A_11 = A[_11]
        B_11 = B[_11]
        for _ in range(5):
            A[_11] += B[:p-1, :p-1]
        #     A[_11] += np.inner(v, A_11 @ v)*_outer_prod(v, v)

        # A[_12] = np.inner(v, B_11 @ v)*w
        # A[_21] = np.inner(v, B_11 @ v)*w
        # A[_22] = np.inner(w, B_11 @ w)
        ed = time.time()
        speed_square_and_insert[i, j] = ed-st

        # st = time.time()
        # A[_11] = np.inner(v, A[_11] @ v)*_outer_prod(v, v)
        # A[_12] = v
        # A[_21] = v
        # A[_22] = 2.0
        # ed = time.time()
        # speed_square_and_insert_with_access[i, j] = ed-st

        # D = np.zeros_like(A)
        # st = time.time()
        # D[_11] = np.inner(v, A_11 @ v)*_outer_prod(v, v)
        # D[_12] = v
        # D[_21] = v
        # D[_22] = 2.0
        # A = D - A
        # ed = time.time()
        # speed_square_and_insert_with_perturb[i, j] = ed-st
        print('p = {} done'.format(p))

    print('repeat {} done'.format(i))
# %%
cmap = plt.cm.get_cmap('tab10')
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for i, (speed, name) in enumerate(zip(
        [
        # speed_mat_prod,
        # speed_square,
        speed_square_and_insert,
        # speed_square_and_insert_with_perturb,
        # speed_square_and_insert_with_access
        ],
        [
        # "B A B",
        # "<v, A_11v> v v.T",
        "Rang 2 insert",
        # "A[_11] = <v, A[_11]v> v v.T,\nA[_12] = ..."
        ])):
    slope_, _ = slope(np.log2(all_p), np.log2(speed.mean(0)))
    ax.plot(all_p, speed.mean(0),
            marker='o', label=name + ' (slope = {:.2f})'.format(slope_), lw=2, color=cmap(i))
    ax.fill_between(all_p, speed.mean(0)+1.96*np.std(speed, axis=0),
                    speed.mean(0)-1.96*np.std(speed, axis=0),
                    alpha=0.4, facecolor=cmap(i))
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid()
ax.legend()

# %%
