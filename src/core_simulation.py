import numpy as np
from numba import jit, prange

@jit(nopython=True, fastmath=True, parallel=True)
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


@jit(nopython=True, fastmath=True,  parallel=False)
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
        #estrategia = np.zeros(total_jog)
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

#Versão MAIS eficiente (nível ideal)
#Atualizar contadores durante a simulação
# mantém Nc, Nd, Np globais e atualiza quando um jogador muda estratégia
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

@jit(nopython=True)
def calc_payoff_por_estrategia(estrategia, payoff, total_jog):
    soma = np.zeros(3)
    cont = np.zeros(3)

    for i in range(total_jog):
        e = estrategia[i]
        soma[e] += payoff[i]
        cont[e] += 1

    medias = np.zeros(3)

    for i in range(3):
        if cont[i] > 0:
            medias[i] = soma[i] / cont[i]
        else:
            medias[i] = 0.0

    return medias

@jit(nopython=True)
def calc_frac_and_payoff(estrategia, payoff, total_jog):
    soma = np.zeros(3)
    count = np.zeros(3)
    c = d = p = 0

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

@jit(nopython=True, fastmath=True,  parallel=False)
def prob_flip(var_pay, k):
    if var_pay > 0: # todo, evita exploções???
        return 1.0 / (1.0 + np.exp(-var_pay / k))
    else:
        exp_val = np.exp(var_pay / k)
        return exp_val / (1.0 + exp_val)
    # return 1 / (1 + np.exp(-var_pay / k))


@jit(nopython=True, fastmath=True,  parallel=False)
def public_good_benefit(C, D, P, params):
    #    L = r*(Nc+Np)/G
    #    L = (r * (Nc + Np) - sigma * Np) / G
    #     L = r * (Nc + Np - sigma * Np) / G
    r = params[0]
    G = params[1]
    c = params[2]
    sigma = params[3]
    return float(r * c * (C + P - sigma * P) / G)

@jit(nopython=True, fastmath=True,  parallel=False)
def conta_estrat(sitio, viz, estrategia):
    vec_estrat = np.zeros(3, dtype=np.int32)  # (C,D,P)
    for j in range(4):  # Conta quantas estratégias existem no grupo centrado no sítio
        vec_estrat[estrategia[viz[sitio, j]]] += 1
    vec_estrat[estrategia[sitio]] += 1
    return vec_estrat

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



# todo olhar!otimizar contagem de fração e
#todo  payoffs de maneira sequencial, so atualiza dentro do if de estratégia ter sido mudada, mas
# isso é compleeeexo, entao por hora deixa a cada inicio de passo de monte-carlo, varremos tudo
@jit(nopython=True, fastmath=True,  parallel=False)
def atualiza_total_estrat(estrategia, payoff, viz, params, total_jog):
    k = params[4]
#    list_atual = np.random.randint(0, total_jog, size=total_jog)  # lista de jogadores aleatórios para um  MCS
#    list_sorteio = np.random.randint(0, 4, size=total_jog)
#    list_prob = np.random.random(size=total_jog)
    for cont in range(0, total_jog):

        atual = np.random.randint(0, total_jog) # list_atual[cont]                       # list_atual[cont]  # np.random.randint(0, total_jog)
#        pay_atual = calcula_payoff(atual, estrategia, viz, params)
        vizsorteado = viz[atual, np.random.randint(0, 4)]  # sorteio de um vizinho aleatório de 0 a 3 #random.randint(0, 3)
#        pay_viz = calcula_payoff(vizsorteado, estrategia, viz, params)
        if estrategia[atual] != estrategia[vizsorteado]:
            pay_atual = payoff[atual]
            pay_viz = payoff[vizsorteado]
            var_pay = pay_viz - pay_atual
            chance_muda = prob_flip(var_pay, k)  # Probabilidade de fermi
            if np.random.random()<chance_muda:
                estrategia[atual] = estrategia[vizsorteado]  # mudança da estratégia do sítio central
                atualiza_payoff_local(atual, estrategia, payoff, viz, params)  # primeiros vizinhos
    #            atualiza_payoff_local_extra(atual, estrategia, payoff, viz, params) #segundos vizinhos



@jit(nopython=True)
def atualiza_payoff_local(atual, estrategia, payoff, viz, params):
    # pega todos os afetados
    for i in range(5):
        temp = atual if i == 0 else viz[atual, i-1]
        payoff[temp] = calcula_payoff(temp, estrategia, viz, params)

@jit(nopython=True)
def atualiza_payoff_local_extra(atual, estrategia, payoff, viz, params):
    # raio 0 e 1
    for i in range(5):
        centro = atual if i == 0 else viz[atual, i-1]
        payoff[centro] = calcula_payoff(centro, estrategia, viz, params)

        # raio 2
        for j in range(4):
            v2 = viz[centro, j]
            payoff[v2] = calcula_payoff(v2, estrategia, viz, params)

@jit(nopython=True)
def mean_payoff(payoff):
    s = 0.0
    for i in range(len(payoff)):
        s += payoff[i]
    return s / len(payoff)


def monte_carlo_single_worker(params, total_jog, total_passos, L, seed): #gpt feito mais estranho ainda
    np.random.seed(seed)

    viz = np.zeros((total_jog, 4), dtype=np.int32)
    inicia_vizinhos(viz, total_jog, L)

    return monte_carlo_single(viz, params, total_jog, total_passos, L, seed)

def monte_carlo_single(viz, params, total_jog, total_passos, L, seed): #gpt feito, estranho

    estrategia = np.zeros(total_jog, dtype=np.int32)
    payoff = np.zeros(total_jog)

    estrat_t = np.zeros((3, total_passos))
    payavg_t = np.zeros((3, total_passos))

    inicia_estrategias(estrategia, total_jog, L, params[6])

    for i in range(total_jog):
        payoff[i] = calcula_payoff(i, estrategia, viz, params)

    for passo in range(total_passos):
        frac, pay = calc_frac_and_payoff(estrategia, payoff, total_jog)

        estrat_t[:, passo] = frac
        payavg_t[:, passo] = pay

        atualiza_total_estrat(estrategia, payoff, viz, params, total_jog)

    return estrat_t, payavg_t

#@jit(nopython=True, fastmath=True,  parallel=False)
def monte_carlo(amo, viz, params, estrategia, payoff, estrat_t, payavg_t, total_jog, total_passos, L):
    # Inicio das estratégias para amostra
    cond_ini = params[6]
    inicia_estrategias(estrategia, total_jog, L, cond_ini)
    # inicia os payoffs de cada sítio
    for i in range(total_jog):
        payoff[i] = calcula_payoff(i, estrategia, viz, params)


    # Começo da simulação de Monte-Carlo
    for passo_atual in range(0, total_passos):
        # calcular todas estatísticas da população no passo atual
#        cfrac, dfrac, pfrac = calc_fracs(estrategia, total_jog)
#        estrat_t[0, amo, passo_atual] = cfrac
#        estrat_t[1, amo, passo_atual] = dfrac
#        estrat_t[2, amo, passo_atual] = pfrac
#        pay_strat = calc_payoff_por_estrategia(estrategia, payoff, total_jog)
#        payavg_t[0, amo, passo_atual] = pay_strat[0]  # C
#        payavg_t[1, amo, passo_atual] = pay_strat[1]  # D
#        payavg_t[2, amo, passo_atual] = pay_strat[2]  # P
        # calculo total direto!!!
        frac, pay_strat = calc_frac_and_payoff(estrategia, payoff, total_jog)
        for count2 in range(3):
            payavg_t[count2, amo, passo_atual] = pay_strat[count2]  # C D P
            estrat_t[count2, amo, passo_atual] = frac[count2]


        # Etapa de atualização da estratégia
        atualiza_total_estrat(estrategia, payoff, viz, params, total_jog)  # atualiza cada rede individualmente

#    if creat_snapshot:
#        plt.figure(amo)
#        plt.colorbar(ticks=[0, 1, 2, 3])
    return estrat_t, payavg_t


# =========================
# MULTI-AMOSTRAS
# =========================

def run_simulation(L, amostras, total_passos, params):
    total_jog = L * L
    viz = np.zeros((total_jog, 4), dtype=np.int32)

    inicia_vizinhos(viz, total_jog, L)
    estrat_t = np.zeros((3, amostras, total_passos))
    payavg_t = np.zeros((3, amostras, total_passos))

    estrategia = np.zeros(total_jog, dtype=np.int32)
    payoff = np.zeros(total_jog)

    for amo in range(amostras):
        estrategia[:] = 0
        payoff[:] = 0

        estrat_t, payavg_t = monte_carlo(
            amo, viz, params,
            estrategia, payoff,
            estrat_t, payavg_t,
            total_jog, total_passos, L
        )

    estrat_medio = np.mean(estrat_t, axis=1)
    payavg_medio = np.mean(payavg_t, axis=1)

    return estrat_t, estrat_medio, payavg_t, payavg_medio