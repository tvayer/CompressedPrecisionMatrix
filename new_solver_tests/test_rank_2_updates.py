# %%
import matplotlib.pyplot as plt
import numpy as np
import time


def _quad_prod(A, x, y):
    return (x * (A @ y)).sum()


def _outer_prod(x, y):
    return np.outer(x, y)


def return_blocks(p, col):
    indices = np.arange(p)
    indices_minus_col = np.concatenate(
        [indices[:col], indices[col + 1:]])
    _11 = indices_minus_col[:, None], indices_minus_col[None]
    _12 = indices_minus_col, col
    _21 = col, indices_minus_col
    _22 = col, col
    return _11, _12, _21, _22


def slope(x, y, n_points=4):
    if len(x) < n_points or len(y) < n_points:
        raise ValueError("Il faut au moins {} points pour calculer la pente.".format(n_points))
    x_dernier = x[-n_points:]
    y_dernier = y[-n_points:]
    slope, intercept = np.polyfit(x_dernier, y_dernier, 1)
    return slope, intercept

# update covariance and its inverse


def update_Theta(Theta, theta_22_new, theta_12_new, _11, _22, _21, _12):
    Theta_new = Theta.copy()
    Theta_new[_22] = theta_22_new
    Theta_new[_12] = theta_12_new
    Theta_new[_21] = theta_12_new
    return Theta_new


def update_W(W, theta_22_new, theta_12_new, _11, _22, _21, _12):
    inv_Theta_11 = W[_11] - (1.0 / W[_22]) * _outer_prod(W[_12], W[_12])

    w_22_new = (
        1.0 / (theta_22_new - _quad_prod(inv_Theta_11, theta_12_new, theta_12_new)))
    w_12_new = - (w_22_new) * (inv_Theta_11 @ theta_12_new)
    W_11_new = inv_Theta_11 + (1.0/w_22_new) * _outer_prod(w_12_new, w_12_new)

    W_new = W.copy()
    W_new[_22] = w_22_new
    W_new[_12] = w_12_new
    W_new[_21] = w_12_new
    W_new[_11] = W_11_new
    return W_new


def update_only_C(C, W_new, W, S, _11, _22, _21, _12):

    log = {}
    log['matmul'] = 0
    log['outer'] = 0
    log['inner'] = 0
    log['sum'] = 0
    log['init'] = 0

    st = time.time()
    W_11 = W[_11]
    W_12 = W[_12]
    W_22 = W[_22]

    W_11_new = W_new[_11]
    W_12_new = W_new[_12]
    W_22_new = W_new[_22]

    S_11 = S[_11]
    S_12 = S[_12]
    S_22 = S[_22]
    log['init'] = time.time() - st

    st = time.time()
    S_11_W_12_new = S_11 @ W_12_new
    S_11_W_12 = S_11 @ W_12
    log['matmul'] += time.time()-st

    # p = C.shape[0]
    st = time.time()
    D_12 = _outer_prod(W_12, W_12 / 2.0)
    D_12_new_new = _outer_prod(W_12_new, W_12_new / 2.0)
    D_12_new = _outer_prod(W_12_new, W_12)
    log['outer'] += time.time() - st

    st = time.time()
    q_12 = np.inner(S_11_W_12, W_12)
    q_12_new_new = np.inner(S_11_W_12_new, W_12_new)
    q_12_new = np.inner(S_11_W_12_new, W_12)
    log['inner'] += time.time()-st

    st = time.time()
    W_12_div_W_22 = W_12 / W_22
    W_new_12_div_W_new_22 = W_12_new / W_22_new
    log['sum'] += time.time() - st

    st = time.time()
    W_11_S_11_W_12_new = W_11 @ S_11_W_12_new
    W_11_S_11_W_12 = W_11 @ S_11_W_12
    log['matmul'] += time.time()-st

    st = time.time()
    tmp1 = _outer_prod(W_new_12_div_W_new_22, W_11_S_11_W_12_new)
    tmp12 = _outer_prod(W_12_div_W_22, W_11_S_11_W_12)
    # print(f'{tmp12.shape=}')
    # print(f'{tmp1.shape=}')
    # print(f'{W_11_S_11_W_12.shape=}')
    # raise prout
    log['outer'] += time.time()-st

    st = time.time()
    tmp1 -= tmp12
    tmp2 = (q_12 / (W_22**2)) * D_12 + (q_12_new_new / (W_22_new**2)) * D_12_new_new
    tmp2 -= (q_12_new / (W_22 * W_22_new)) * D_12_new
    log['sum'] += time.time() - st

    st = time.time()
    tmp3 = _outer_prod(W_12_new, W_11_new @ S_12)
    tmp4 = -_outer_prod(W_12, W_11 @ S_12)
    log['outer'] += time.time()-st

    st = time.time()
    tmp5 = S_22 * (D_12_new_new - D_12)
    tmp_sum = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
    tmp_sum += tmp_sum.T  # Symmetrize tmp_sum
    C[_11] += tmp_sum
    log['sum'] += time.time() - st

    st = time.time()
    S_12_W_new_12 = np.inner(S_12, W_12_new)
    log['inner'] += time.time()-st

    st = time.time()
    C[_22] = q_12_new_new + 2 * W_22_new * S_12_W_new_12 + (W_22_new**2) * S_22
    log['sum'] += time.time() - st

    st = time.time()
    W_new_11_S_11_W_new_12 = W_11_new @ S_11_W_12_new
    W_new_11_S_12 = W_11_new @ S_12
    log['matmul'] += time.time()-st

    st = time.time()
    S_12_W_new_12 = np.inner(S_12, W_12_new)
    log['inner'] += time.time()-st

    st = time.time()
    C_12 = W_new_11_S_11_W_new_12 + S_12_W_new_12 * W_12_new + W_22_new * (W_new_11_S_12 + S_22 * W_12_new)
    C[_12] = C_12
    C[_21] = C_12
    log['sum'] += time.time() - st

    return C, log

# def update_only_C(C, W_new, W, S, _11, _22, _21, _12):

#     W_11 = W[_11]
#     W_12 = W[_12]
#     W_22 = W[_22]

#     W_11_new = W_new[_11]
#     W_12_new = W_new[_12]
#     W_22_new = W_new[_22]

#     S_11 = S[_11]
#     S_12 = S[_12]
#     S_22 = S[_22]

#     S_11_W_12_new = S_11 @ W_12_new
#     S_11_W_12 = S_11 @ W_12

#     # p = C.shape[0]
#     D_12 = _outer_prod(W_12, W_12) / 2.0
#     D_12_new_new = _outer_prod(W_12_new, W_12_new) / 2.0
#     D_12_new = _outer_prod(W_12_new, W_12)

#     q_12 = np.inner(S_11_W_12, W_12)
#     q_12_new_new = np.inner(S_11_W_12_new, W_12_new)
#     q_12_new = np.inner(S_11_W_12_new, W_12)

#     W_12_div_W_22 = W_12 / W_22
#     W_new_12_div_W_new_22 = W_12_new / W_22_new

#     W_11_S_11_W_12_new = W_11 @ S_11_W_12_new
#     W_11_S_11_W_12 = W_11 @ S_11_W_12

#     tmp1 = _outer_prod(W_new_12_div_W_new_22, W_11_S_11_W_12_new) \
#         - _outer_prod(W_12_div_W_22, W_11_S_11_W_12)

#     tmp2 = (q_12 / (W_22**2)) * D_12 \
#         + (q_12_new_new / (W_22_new**2)) * D_12_new_new \
#         - (q_12_new / (W_22 * W_22_new)) * D_12_new

#     tmp3 = _outer_prod(W_12_new, W_11_new @ S_12)
#     tmp4 = -_outer_prod(W_12, W_11 @ S_12)

#     tmp5 = S_22 * (D_12_new_new - D_12)
#     tmp_sum = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
#     tmp_sum += tmp_sum.T  # Symmetrize tmp_sum
#     C[_11] += tmp_sum

#     S_12_W_new_12 = np.inner(S_12, W_12_new)

#     C[_22] = q_12_new_new + 2 * W_22_new * S_12_W_new_12 + (W_22_new**2) * S_22

#     W_new_11_S_11_W_new_12 = W_11_new @ S_11_W_12_new
#     W_new_11_S_12 = W_11_new @ S_12

#     S_12_W_new_12 = np.inner(S_12, W_12_new)
#     C_12 = W_new_11_S_11_W_new_12 + S_12_W_new_12 * W_12_new + W_22_new * (W_new_11_S_12 + S_22 * W_12_new)
#     C[_12] = C_12
#     C[_21] = C_12

#     return C


# %%
all_p = [500, 1000, 2000, 3000, 5000, 10000]
# all_p = [5]
all_equal = []
n = 100
repeat = 3
rang_2_update = np.zeros((repeat, len(all_p)))
mat_prod_update = np.zeros((repeat, len(all_p)))

summ = np.zeros((repeat, len(all_p)))
outer = np.zeros((repeat, len(all_p)))
init = np.zeros((repeat, len(all_p)))
matmul = np.zeros((repeat, len(all_p)))

for i in range(repeat):
    for j, p in enumerate(all_p):
        X = np.random.randn(n, p)
        Y = np.random.randn(n, p)
        S = Y.T @ Y + 2*np.eye(p)
        Theta = X.T @ X + np.eye(p)
        W = np.eye(p) - X.T @ np.linalg.inv(np.eye(n) + X @ X.T) @ X  # woodbury for fast inverse
        C = W @ S @ W
        _11, _12, _21, _22 = return_blocks(p, 0)

        theta_12_new = np.random.randn(p-1)
        theta_22_new = 2.0
        W_new = update_W(W, theta_22_new, theta_12_new, _11, _22, _21, _12)

        st = time.time()
        C_new_our, log = update_only_C(C, W_new, W, S, _11, _22, _21, _12)
        ed = time.time()
        rang_2_update[i, j] = ed-st

        summ[i, j] = log['sum']
        outer[i, j] = log['outer']
        init[i, j] = log['init']
        matmul[i, j] = log['matmul']

        st = time.time()
        C_new_mat_prod = W_new @ S @ W_new
        ed = time.time()
        mat_prod_update[i, j] = ed-st
        all_equal.append(np.linalg.norm(C_new_mat_prod - C_new_our, ord=np.inf))

        # st = time.time()
        # # v = np.random.randn(p-1, p-1) @ theta_12_new
        # _ = _outer_prod(theta_12_new, np.random.randn(p-1, p-1) @ theta_12_new)
        # ed = time.time()
        # res_outer.append(ed-st)
        print('p = {} done '.format(p))
    print('repeat {} done '.format(i))
# %%
print(all_equal)
# %%
fs = 12
cmap = plt.cm.get_cmap('tab10')
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for i, (speed, name) in enumerate(zip(
        [
        rang_2_update,
        mat_prod_update,
        summ,
        outer,
        init
        ],
        [
        # "B A B",
        "Rang 2",
        "Matrix product",
        "sum",
        "outer",
        "init"
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
ax.set_xlabel('p', fontsize=fs)
ax.set_ylabel('time (in sec.)', fontsize=fs)

# %%
# def optimized_update_B(B, W_new, W, S, _11, _22, _21, _12):
#     B_new = B.copy()

#     # Calculs partagés
#     S_11_W_12 = S[_11].T @ W[_12]
#     S_11_W_new12 = S[_11].T @ W_new[_12]

#     # Mise à jour de B_new[_22], B_new[_12], et B_new[_21]
#     B_new[_22] = (W_new[_12] * S[_12]).sum() + W_new[_22] * S[_22]
#     B_new[_12] = W_new[_11] @ S[_12] + S[_22] * W_new[_12]
#     B_new[_21] = (W_new[_12][None, :] @ S[_11]).ravel() + W_new[_22] * S[_12]

#     # Optimisation des termes tmp1, tmp2, tmp3, et _outer_prod(W_new[_12], S[_12])
#     tmp1 = _outer_prod(W[_12], S[_12])
#     tmp2 = (1.0 / W[_22]) * _outer_prod(W[_12], S_11_W_12)
#     tmp3 = (1.0 / W_new[_22]) * _outer_prod(W_new[_12], S_11_W_new12)
#     tmp4 = _outer_prod(W_new[_12], S[_12])

#     # Calcul de B_new[_11] avec optimisation
#     B_new[_11] = B[_11] - tmp1 - tmp2 + tmp3 + tmp4

#     return B_new


# def optimized_update_C(C, B_new, W_new, W, S, _11, _22, _21, _12):
#     C_new = C.copy()

#     # Mise à jour de C_new[_22], C_new[_12], et C_new[_21]
#     C_new[_22] = (B_new[_21] * W_new[_12]).sum() + B_new[_22] * W_new[_22]
#     C_new[_12] = B_new[_11] @ W_new[_12] + W_new[_22] * B_new[_12]
#     C_new[_21] = B_new[_21] @ W_new[_11] + B_new[_22] * W_new[_12]

#     # Calculs partagés pour réduire les calculs redondants
#     W_11_S_12 = W[_11].T @ S[_12]
#     S_11_W_12 = S[_11].T @ W[_12]
#     S_11_W_new12 = S[_11].T @ W_new[_12]

#     # Optimisation des termes tmp11, tmp12, tmp13, et tmp14
#     tmp11 = _outer_prod(W[_12], W_11_S_12)
#     tmp12 = (1.0 / W[_22]) * _outer_prod(W[_12], W[_11].T @ S_11_W_12)
#     tmp13 = (1.0 / W_new[_22]) * _outer_prod(W_new[_12], W[_11].T @ S_11_W_new12)
#     tmp14 = _outer_prod(W_new[_12], W_11_S_12)

#     # Calcul de tmp1 optimisé
#     tmp1 = C[_11] - _outer_prod(B[_12], W[_12]) - tmp11 - tmp12 + tmp13 + tmp14

#     # Calcul des termes tmp2, tmp3, et tmp4
#     tmp2 = (1.0 / W[_22]) * _outer_prod(B_new[_11] @ W[_12], W[_12])
#     tmp3 = (1.0 / W_new[_22]) * _outer_prod(B_new[_11] @ W_new[_12], W_new[_12])
#     tmp4 = _outer_prod(B_new[_12], W_new[_12])

#     # Calcul final de C_new[_11]
#     C_new[_11] = tmp1 - tmp2 + tmp3 + tmp4

#     return C_new

# def update_C(C, B_new, W_new, W, S, _11, _22, _21, _12):
#     C_new = C.copy()
#     C_new[_22] = (B_new[_21] * W_new[_12]).sum() + B_new[_22] * W_new[_22]  # good
#     C_new[_12] = B_new[_11] @ W_new[_12] + W_new[_22] * B_new[_12]  # good
#     C_new[_21] = B_new[_21] @ W_new[_11] + B_new[_22] * W_new[_12]  # good

#     tmp11 = _outer_prod(W[_12], W[_11].T @ S[_12])
#     tmp12 = (1.0 / W[_22]) * _outer_prod(W[_12], W[_11].T @ (S[_11].T @ W[_12]))
#     tmp13 = (1.0 / W_new[_22]) * _outer_prod(W_new[_12], W[_11].T @ (S[_11].T @ W_new[_12]))
#     tmp14 = _outer_prod(W_new[_12], W[_11].T @ S[_12])
#     tmp1 = C[_11] - _outer_prod(B[_12], W[_12]) - tmp11 - tmp12 + tmp13 + tmp14

#     tmp2 = (1.0 / W[_22]) * _outer_prod(B_new[_11] @ W[_12], W[_12])
#     tmp3 = (1.0 / W_new[_22]) * _outer_prod(B_new[_11] @ W_new[_12], W_new[_12])
#     tmp4 = _outer_prod(B_new[_12], W_new[_12])
#     C_new[_11] = tmp1 - tmp2 + tmp3 + tmp4
#     # good but d^3

#     return C_new

# def update_only_C(C, W_new, W, S, _11, _22, _21, _12):

#     S_11_W_12_new = S[_11] @ W_new[_12]
#     S_11_W_12 = S[_11] @ W[_12]

#     D_12 = _outer_prod(W[_12], W[_12]) / 2.0
#     D_12_new_new = _outer_prod(W_new[_12], W_new[_12]) / 2.0
#     D_12_new = _outer_prod(W_new[_12], W[_12])
#     # D_12_new += D_12_new.T

#     q_12 = np.dot(S_11_W_12, W[_12])
#     q_12_new_new = np.dot(S_11_W_12_new, W_new[_12])
#     q_12_new = np.dot(S_11_W_12_new, W[_12])

#     tmp1 = -_outer_prod(W[_12] / W[_22], W[_11] @ S_11_W_12)
#     tmp1 += _outer_prod(W_new[_12] / W_new[_22], W[_11] @ S_11_W_12_new)
#     # tmp1 += tmp1.T

#     tmp2 = (q_12 / (W[_22]**2))*D_12
#     tmp2 += (q_12_new_new / (W_new[_22]**2))*D_12_new_new
#     tmp2 -= (q_12_new / (W[_22]*W_new[_22]))*D_12_new

#     tmp3 = _outer_prod(W_new[_12], W_new[_11] @ S[_12])
#     # tmp3 += tmp3.T

#     tmp4 = -_outer_prod(W[_12], W[_11] @ S[_12])
#     # tmp4 += tmp4.T

#     tmp5 = S[_22]*(D_12_new_new - D_12)

#     tmp_sum = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
#     tmp_sum += tmp_sum.T
#     C[_11] += tmp_sum

#     C[_22] = q_12_new_new + 2*W_new[_22]*np.dot(S[_12], W_new[_12]) + (W_new[_22]**2)*S[_22]

#     C[_12] = W_new[_11] @ S_11_W_12_new + np.dot(S[_12], W_new[_12])*W_new[_12]
#     C[_12] += W_new[_22]*(W_new[_11] @ S[_12] + S[_22]*W_new[_12])
#     C[_21] = C[_12]

#     return C

# def update_only_C(C, W_new, W, S, _11, _22, _21, _12):

#     S_11_W_12_new = S[_11] @ W_new[_12]
#     S_11_W_12 = S[_11] @ W[_12]
#     p = C.shape[0]

#     v = np.concatenate((W[_12], W_new[_12]))
#     V = _outer_prod(v, v)

#     D_12 = V[:p-1, :p-1] / 2.0
#     D_12_new_new = V[p-1:, p-1:] / 2.0
#     D_12_new = V[p-1:, :p-1]

#     q_12 = np.dot(S_11_W_12, W[_12])
#     q_12_new_new = np.dot(S_11_W_12_new, W_new[_12])
#     q_12_new = np.dot(S_11_W_12_new, W[_12])

#     W_12_div_W_22 = W[_12] / W[_22]
#     W_new_12_div_W_new_22 = W_new[_12] / W_new[_22]
#     tmp1 = _outer_prod(W_new_12_div_W_new_22, W[_11] @ S_11_W_12_new) \
#         - _outer_prod(W_12_div_W_22, W[_11] @ S_11_W_12)

#     tmp2 = (q_12 / (W[_22]**2)) * D_12 \
#         + (q_12_new_new / (W_new[_22]**2)) * D_12_new_new \
#         - (q_12_new / (W[_22] * W_new[_22])) * D_12_new

#     tmp3 = _outer_prod(W_new[_12], W_new[_11] @ S[_12])
#     tmp4 = -_outer_prod(W[_12], W[_11] @ S[_12])

#     tmp5 = S[_22] * (D_12_new_new - D_12)

#     tmp_sum = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
#     tmp_sum += tmp_sum.T  # Symmetrize tmp_sum
#     C[_11] += tmp_sum

#     C[_22] = q_12_new_new + 2 * W_new[_22] * np.dot(S[_12], W_new[_12]) + (W_new[_22]**2) * S[_22]

#     C[_12] = W_new[_11] @ S_11_W_12_new + np.dot(S[_12], W_new[_12]) * W_new[_12]
#     C[_12] += W_new[_22] * (W_new[_11] @ S[_12] + S[_22] * W_new[_12])
#     C[_21] = C[_12]

#     return C
