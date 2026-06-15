import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def seed_numba(seed):
    np.random.seed(seed)


@jit(nopython=True, fastmath=True, parallel=False, cache=True)
def inicia_vizinhos(viz, total_jog, L):
    # Inicialização das estratégias e definição de vizinhos
    for jogador_atual in prange(total_jog):  # Lembrete, for in range vai até total_jog-1!!
        viz[jogador_atual, 1] = jogador_atual + 1  # Vizinho direito=1
        viz[jogador_atual, 3] = jogador_atual - 1  # Vizinho esquerdo=3
        viz[jogador_atual, 2] = jogador_atual + L  # Vizinho de baixo=2
        viz[jogador_atual, 0] = jogador_atual - L  # Vizinho de cima=0
        # definindo condições de contorno periódicas na rede quadrada
        if (jogador_atual - L + 1) % L == 0:  # definindo coluna direita
            viz[jogador_atual, 1] = jogador_atual + 1 - L  # definindo vizinho direito
        if jogador_atual % L == 0 or jogador_atual == 0:  # definindo coluna esquerda
            viz[jogador_atual, 3] = jogador_atual + L - 1  # definindo vizinho esquerdo
        if L > jogador_atual >= 0:
            viz[jogador_atual, 0] = jogador_atual + L * L - L
        if L * L > jogador_atual >= L * L - L:
            viz[jogador_atual, 2] = jogador_atual - (L * L - L)


@jit(nopython=True, fastmath=True,  parallel=False, cache=True)
def inicia_estrategias(estrategia, total_jog, L, cond_ini):
    # Inicialização das estratégias
    if cond_ini == 0:
        for jogador_atual in range(total_jog):  # Lembete, for in range vai até total_jog-1!!
            estrategia[jogador_atual] = np.random.randint(3)  # Estado inicial do jogador como aleatório
    elif cond_ini == 1:
        for i in range(0, L):
            for j in range(0, L):
                if i <= L//3:
                    estrategia[i * L + j] = 0
                elif L // 3 < i and i <= 2 * L // 3:
                        estrategia[i * L + j] = 1
                elif  2 * L//3 < i:
                    estrategia[i * L + j] = 2
    elif cond_ini == 2:
        tgangle=0.75
        for i in range(0, L):
            for j in range(0, L):
                estrategia[j * L + i] = 0
        for i in range(0, L):
            for j in range(0, L):
                if i < (L - tgangle*j):
                    estrategia[j * L + i] = 1
        for i in range(0, L):
            for j in range(0, L):
                if i < tgangle*j  and i < L//2:
                    estrategia[j * L + i] = 2
    elif cond_ini == 4:
        for jogador_atual in range(total_jog):  # Lembete, for in range vai até total_jog-1!!
            estrategia[jogador_atual] = np.random.randint(2)  # Estado inicial do jogador como aleatório

@jit(nopython=True, cache=True)
def calc_frac_and_payoff(estrategia, payoff, total_jog):
    soma = np.zeros(3)
    count = np.zeros(3)

    for i in range(total_jog):
        e = estrategia[i]
        soma[e] += payoff[i]
        count[e] += 1
    frac = count / total_jog

    media = np.zeros(3)
    for i in range(3):
        if count[i] > 0:
            media[i] = soma[i] / count[i]

    return frac, media

@jit(nopython=True, fastmath=True,  parallel=False, cache=True)
def prob_flip(var_pay, k):
    if var_pay > 0:
        return 1.0 / (1.0 + np.exp(-var_pay / k))
    else:
        exp_val = np.exp(var_pay / k)
        return exp_val / (1.0 + exp_val)


@jit(nopython=True, fastmath=True,  parallel=False, cache=True)
def public_good_benefit(C, D, P, params):
    r = params[0]
    G = params[1]
    c = params[2]
    sigma = params[3]
    return float(r * c * (C + P - sigma * P) / G)

@jit(nopython=True, fastmath=True,  parallel=False, cache=True)
def conta_estrat(sitio, viz, estrategia):
    vec_estrat = np.zeros(3, dtype=np.int32)  # (C,D,P)
    for j in range(4):  # Conta quantas estratégias existem no grupo centrado no sítio
        vec_estrat[estrategia[viz[sitio, j]]] += 1
    vec_estrat[estrategia[sitio]] += 1
    return vec_estrat

@jit(nopython=True, cache=True)
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

@jit(nopython=True, fastmath=True,  parallel=False, cache=True)
def atualiza_total_estrat(estrategia, payoff, viz, params, total_jog):
    k = params[4]
    activity = 0
    for cont in range(0, total_jog):

        atual = np.random.randint(0, total_jog)
        vizsorteado = viz[atual, np.random.randint(0, 4)]  # sorteio de um vizinho aleatório de 0 a 3 #random.randint(0, 3)
        if estrategia[atual] != estrategia[vizsorteado]:
            pay_atual = payoff[atual]
            pay_viz = payoff[vizsorteado]
            var_pay = pay_viz - pay_atual
            chance_muda = prob_flip(var_pay, k)  # Probabilidade de fermi
            if np.random.random()<chance_muda:
                estrategia[atual] = estrategia[vizsorteado]  # mudança da estratégia do sítio central
                activity += 1
                atualiza_payoff_local_extra(atual, estrategia, payoff, viz, params) #segundos vizinhos

    return activity


@jit(nopython=True, cache=True)
def atualiza_payoff_local_extra(atual, estrategia, payoff, viz, params):
    # raio 0 e 1
    for i in range(5):
        centro = atual if i == 0 else viz[atual, i-1]
        payoff[centro] = calcula_payoff(centro, estrategia, viz, params)

        # raio 2
        for j in range(4):
            v2 = viz[centro, j]
            payoff[v2] = calcula_payoff(v2, estrategia, viz, params)

def monte_carlo_single_worker(params, total_jog, total_passos, L, seed, absorbing_window=0):
    viz = np.zeros((total_jog, 4), dtype=np.int32)
    inicia_vizinhos(viz, total_jog, L)

    return monte_carlo_single(viz, params, total_jog, total_passos, L, seed, absorbing_window=absorbing_window)

def monte_carlo_single(viz, params, total_jog, total_passos, L, seed, callback=None, absorbing_window=0):
    if seed is not None:
        seed_numba(seed)

    estrategia = np.zeros(total_jog, dtype=np.int32)
    payoff = np.zeros(total_jog)

    estrat_t = np.zeros((3, total_passos))
    payavg_t = np.zeros((3, total_passos))
    activity_t = np.zeros(total_passos)
    absorbed_at = total_passos

    inicia_estrategias(estrategia, total_jog, L, params[6])

    for i in range(total_jog):
        payoff[i] = calcula_payoff(i, estrategia, viz, params)

    for passo in range(total_passos):
        frac, pay = calc_frac_and_payoff(estrategia, payoff, total_jog)

        estrat_t[:, passo] = frac
        payavg_t[:, passo] = pay

        # Callback opcional para visualizacao ou diagnostico.
        if callback is not None:
            callback(passo, estrategia, payoff)

        activity_t[passo] = atualiza_total_estrat(estrategia, payoff, viz, params, total_jog) / total_jog

        if absorbing_window > 0 and passo + 1 >= absorbing_window:
            start = passo + 1 - absorbing_window
            if np.sum(activity_t[start:passo + 1]) == 0:
                absorbed_at = passo + 1
                for future in range(passo + 1, total_passos):
                    estrat_t[:, future] = estrat_t[:, passo]
                    payavg_t[:, future] = payavg_t[:, passo]
                    activity_t[future] = 0
                break

    return estrat_t, payavg_t, activity_t, absorbed_at


