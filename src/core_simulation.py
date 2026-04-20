import numpy as np
from numba import jit, prange

# =========================
# INICIALIZAÇÃO
# =========================

@jit(nopython=True, fastmath=True, parallel=False)
def inicia_vizinhos(viz, total_jog, L):
    for jogador_atual in prange(total_jog):
        viz[jogador_atual, 1] = jogador_atual + 1
        viz[jogador_atual, 3] = jogador_atual - 1
        viz[jogador_atual, 2] = jogador_atual + L
        viz[jogador_atual, 0] = jogador_atual - L

        if (jogador_atual - L + 1) % L == 0:
            viz[jogador_atual, 1] = jogador_atual + 1 - L
        if jogador_atual % L == 0:
            viz[jogador_atual, 3] = jogador_atual + L - 1
        if jogador_atual < L:
            viz[jogador_atual, 0] = jogador_atual + L * L - L
        if jogador_atual >= L * L - L:
            viz[jogador_atual, 2] = jogador_atual - (L * L - L)


@jit(nopython=True, fastmath=True)
def inicia_estrategias(estrategia, total_jog, L, cond_ini):
    if cond_ini == 0:
        for i in range(total_jog):
            estrategia[i] = np.random.randint(3)

    elif cond_ini == 1:
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                if i <= L//3:
                    estrategia[idx] = 0
                elif i <= 2 * L // 3:
                    estrategia[idx] = 1
                else:
                    estrategia[idx] = 2

    elif cond_ini == 2:
        tgangle = 0.75
        for i in range(L):
            for j in range(L):
                estrategia[j * L + i] = 0

        for i in range(L):
            for j in range(L):
                if i < (L - tgangle*j):
                    estrategia[j * L + i] = 1

        for i in range(L):
            for j in range(L):
                if i < tgangle*j and i < L//2:
                    estrategia[j * L + i] = 2


# =========================
# DINÂMICA
# =========================

@jit(nopython=True)
def calc_fracs(estrategia, total_jog):
    c = d = p = 0
    for i in range(total_jog):
        if estrategia[i] == 0:
            c += 1
        elif estrategia[i] == 1:
            d += 1
        else:
            p += 1
    return c/total_jog, d/total_jog, p/total_jog


@jit(nopython=True, fastmath=True)
def prob_flip(var_pay, k):
    if var_pay > 0:
        return 1.0 / (1.0 + np.exp(-var_pay / k))
    else:
        exp_val = np.exp(var_pay / k)
        return exp_val / (1.0 + exp_val)


@jit(nopython=True, fastmath=True)
def public_good_benefit(C, D, P, params):
    r, G, c, sigma = params[0], params[1], params[2], params[3]
    return r * c * (C + P - sigma * P) / G


@jit(nopython=True)
def conta_estrat(sitio, viz, estrategia):
    vec = np.zeros(3, dtype=np.int32)
    for j in range(4):
        vec[estrategia[viz[sitio, j]]] += 1
    vec[estrategia[sitio]] += 1
    return vec


@jit(nopython=True)
def calcula_payoff(sitio, estrategia, viz, params):
    c = params[2]
    sigma = params[3]
    alpha = params[5]

    payoff = 0.0
    estrat = estrategia[sitio]

    for i in range(5):
        grupo = sitio if i == 0 else viz[sitio, i-1]
        Nc, Nd, Np = conta_estrat(grupo, viz, estrategia)

        payoff += public_good_benefit(Nc, Nd, Np, params)

        if estrat == 0:
            payoff -= c
            if Nc > 0:
                payoff += alpha * sigma * c * (Np / Nc)

        elif estrat == 2:
            payoff -= c
            if Nc > 0:
                payoff += (1 - alpha) * sigma * c
            else:
                payoff += sigma * c

    return payoff


@jit(nopython=True)
def atualiza_total_estrat(estrategia, viz, params, total_jog):
    k = params[4]

    for _ in range(total_jog):
        atual = np.random.randint(0, total_jog)
        vizinho = viz[atual, np.random.randint(0, 4)]

        pay_atual = calcula_payoff(atual, estrategia, viz, params)
        pay_viz = calcula_payoff(vizinho, estrategia, viz, params)

        if np.random.random() < prob_flip(pay_viz - pay_atual, k):
            estrategia[atual] = estrategia[vizinho]


# =========================
# MONTE CARLO (CORE)
# =========================

def monte_carlo(viz, params, L, total_passos, cond_ini):
    total_jog = L * L

    estrategia = np.zeros(total_jog, dtype=np.int32)
    estrat_t = np.zeros((3, total_passos))

    inicia_estrategias(estrategia, total_jog, L, cond_ini)

    for t in range(total_passos):
        c, d, p = calc_fracs(estrategia, total_jog)
        estrat_t[0, t] = c
        estrat_t[1, t] = d
        estrat_t[2, t] = p

        atualiza_total_estrat(estrategia, viz, params, total_jog)

    return estrat_t


# =========================
# MULTI-AMOSTRAS
# =========================

def run_simulation(L, amostras, total_passos, params, cond_ini):
    total_jog = L * L
    viz = np.zeros((total_jog, 4), dtype=np.int32)

    inicia_vizinhos(viz, total_jog, L)

    estrat_t = np.zeros((3, amostras, total_passos))

    for amo in range(amostras):
        estrat_t[:, amo, :] = monte_carlo(
            viz, params, L, total_passos, cond_ini
        )

    estrat_medio = np.mean(estrat_t, axis=1)

    return estrat_t, estrat_medio