import numpy as np
import matplotlib.pyplot as plt
import time
# import random
from numba import jit, prange
from celluloid import Camera

# Parâmetros da simulação:
L = 100  # tamanho da rede
amostras = 1
total_passos = 100
passos_media = int(0.9 * total_passos)
framerate = 3
passo_filma_inicio = 0
k = 0.1  # Irracionalidade
r = 5  # Multiplication factor
G = 5  # Group Size
c = 1
sigma = 3.9
total_jog = L * L
start_time = time.time()
np.random.seed()


# Gerador da imagem
def generate_imagegrid(imagegrid, estrategia_):
    for i in range(0, L):
        for j in range(0, L):
            imagegrid[i][j] = estrategia_[i + j * L]
            # 0=C,1=D,2=P


@jit(nopython=True, fastmath=True, parallel=False)
def inicia_vizinhos(viz_):
    # Inicialização das estratégias e definição de vizinhos
    for jogador_atual in prange(total_jog):  # Lembrete, for in range vai até total_jog-1!!
        viz_[jogador_atual, 1] = jogador_atual + 1  # Vizinho direito=1
        viz_[jogador_atual, 3] = jogador_atual - 1  # Vizinho esquerdo=3
        viz_[jogador_atual, 2] = jogador_atual + L  # Vizinho de baixo=2
        viz_[jogador_atual, 0] = jogador_atual - L  # Vizinho de cima=0
        # definindo condições de contorno periódicas na rede quadrada
        if (jogador_atual - L + 1) % L == 0:  # definindo coluna direita
            viz_[jogador_atual, 1] = jogador_atual + 1 - L  # definindo vizinho direito
        if jogador_atual % L == 0 or jogador_atual == 0:  # definindo coluna esquerda
            viz_[jogador_atual, 3] = jogador_atual + L - 1  # definindo vizinho esquerdo
        if L > jogador_atual >= 0:
            viz_[jogador_atual, 0] = jogador_atual + L * L - L
        if L * L > jogador_atual >= L * L - L:
            viz_[jogador_atual, 2] = jogador_atual - (L * L - L)


@jit(nopython=True, fastmath=True, parallel=False)
def inicia_estrategias(estrategia_):
    # Inicialização das estratégias
    for jogador_atual in range(total_jog):  # Lembete, for in range vai até total_jog-1!!
        estrategia_[jogador_atual] = np.random.randint(3)  # Estado inicial do jogador como aleatório


@jit(nopython=True, fastmath=True, parallel=False)  # todo Gasto de tempo enorme, como otimizar?
def calc_fracs(amo_, estrat_t_, estrategia_, passo_atual_):
    estrat_t_[0, amo_, passo_atual_] = np.sum(estrategia_ == 0) / total_jog  # C
    estrat_t_[1, amo_, passo_atual_] = np.sum(estrategia_ == 1) / total_jog  # D
    estrat_t_[2, amo_, passo_atual_] = np.sum(estrategia_ == 2) / total_jog  # P


@jit(nopython=True, fastmath=True, parallel=False)
def prob_flip(var_pay):
    if var_pay > 0:  # todo, evita exploções???
        return 1.0 / (1.0 + np.exp(-var_pay / k))
    else:
        exp_val = np.exp(var_pay / k)
        return exp_val / (1.0 + exp_val)
# return 1 / (1 + np.exp(-var_pay / k_))


@jit(nopython=True, fastmath=True, parallel=False)
def public_good_benefit(C, D, P, params_):
    #    L = r*(Nc+Np)/G
    #    L = (r * (Nc + Np) - sigma * Np) / G
    #     L = r * (Nc + Np - sigma * Np) / G
    r, G, c, sigma = params_
    return float(r * c * (C + P - sigma * P) / G)


@jit(nopython=True, fastmath=True, parallel=False)
def conta_estrat(sitio, viz_, estrategia_):
    vec_estrat = np.zeros(3)  # (C,D,P)
    for j in range(4):  # Conta quantas estratégias existem no grupo centrado no sítio
        vec_estrat[estrategia_[viz_[sitio, j]]] += 1
    vec_estrat[estrategia_[sitio]] += 1
    return vec_estrat


@jit(nopython=True, fastmath=True, parallel=False)  # OTIMIZAR!!!!!!
def calcula_payoff(sitio, estrategia_, viz_, params_):
    payoff_sitio = 0
    Nc, Nd, Np = conta_estrat(sitio, viz_, estrategia_)
    payoff_sitio += public_good_benefit(Nc, Nd, Np,
                                        params_)  # matriz_pay_[int(Nc),int(Nd)] # r * (Nc + Np - sigma * Np) / G
    # jogo centrado em cada um dos 4 vizinhos
    for i in range(4):
        sitio_temp = viz_[sitio, i]
        Nc, Nd, Np = conta_estrat(sitio_temp, viz_, estrategia_)
        payoff_sitio += public_good_benefit(Nc, Nd, Np,
                                            params_)  # matriz_pay_[int(Nc),int(Nd)] # r * (Nc + Np - sigma * Np) / G
    if estrategia_[sitio] == 0:
        payoff_sitio = payoff_sitio - 5 * c
    if estrategia_[sitio] == 2:
        payoff_sitio = payoff_sitio - 5 * c + 5 * sigma * c
    return payoff_sitio


@jit(nopython=True, fastmath=True, parallel=False)  # todo olhar!!!
def atualiza_total_estrat(estrategia_, viz_, params_):
    list_atual = np.random.randint(0, total_jog, size=total_jog)  # lista de jogadores aleatórios para um  MCS
    list_sorteio = np.random.randint(0, 4, size=total_jog)
    list_prob = np.random.random(size=total_jog)
    for cont_ in range(0, total_jog):
        atual = list_atual[cont_]  # list_atual[cont_]  # np.random.randint(0, total_jog)
        pay_atual = calcula_payoff(atual, estrategia_, viz_, params_)
        vizsorteado = viz_[atual, list_sorteio[cont_]]  # sorteio de um vizinho aleatório de 0 a 3 #random.randint(0, 3)
        pay_viz = calcula_payoff(vizsorteado, estrategia_, viz_, params_)
        var_pay_ = pay_viz - pay_atual
        prob = list_prob[cont_]  # list_prob[cont_] # random.random()  # probabilidade aleatória
        chance_muda = prob_flip(var_pay_)  # Probabilidade de fermi
        if prob < chance_muda:
            estrategia_[atual] = estrategia_[vizsorteado]  # mudança da estratégia do sítio central


# @jit(nopython=True, fastmath=True,  parallel=False)
def monte_carlo(amo_, viz_, params_, estrategia_, estrat_t_):
    # Inicio das estratégias para amostra
    inicia_estrategias(estrategia_)
    # Começo da simulação de Monte-Carlo
    for passo_atual in range(0, total_passos):
        # Antes de qualquer alteração, calcular todas estatísticas da população no passo atual
        calc_fracs(amo_, estrat_t_, estrategia_, passo_atual)
        # Gerando as imagens da rede
        if passo_atual % framerate == 0 and passo_atual >= passo_filma_inicio:
            generate_imagegrid(imagegrid, estrategia)
            plt.figure(1)
            plt.imshow(imagegrid, vmin=0, vmax=2, cmap='brg', interpolation='none')
            plt.title(f"C=0 D=1 P=2, mcs={passo_atual}")
            plt.savefig('%s.png' % passo_atual)
            camera.snap()
        # Etapa de atualização da estratégia
        atualiza_total_estrat(estrategia_, viz_, params_)  # atualiza cada rede individualmente
    plt.figure(1)
    plt.colorbar(ticks=[0, 1, 2, 3])


# @jit(nopython=True, fastmath=True,  parallel=False) Numba não compatível com impressão aparentemente
def imprime_dados(estrat_medio_t):
    arquivo2 = open("subpop.dat", "w+")
    for cont in range(0, total_passos):
        arquivo2.write(f"{cont} {estrat_medio_t[0, cont]:.4f} {estrat_medio_t[1, cont]:.4f} "
                       f"{estrat_medio_t[2, cont]:.4f} \n")
    print(f"----- {(time.time() - start_time):.4f} seconds or {((time.time() - start_time) / 60.0):.4f} minutes-----")
    arquivo2.close()


def plota_dados(estrat_medio_t_):
    plt.figure(1)
    plt.title('Evolução média das subpopulações')
    plt.xlabel('Número de Passos')
    plt.ylabel(r'$<\rho >$')
    plt.plot(estrat_medio_t_[0, :], label='C')
    plt.plot(estrat_medio_t_[1, :], label='D')
    plt.plot(estrat_medio_t_[2, :], label='P')
    plt.legend()
    plt.show()


# Definição de variáveis do jogo
estrategia = np.zeros(total_jog, dtype=int)
estrat_t = np.zeros((3, amostras, total_passos))
params = np.zeros(4)  # rede,linha,coluna
# Definição de vizinhos
viz = np.zeros((total_jog, 4), dtype=int)  # matriz contendo os vizinhos 0=cima,1=direita,2=baixo,3=esquerda
# Variáveis da imagem
fig = plt.figure()
imagegrid = np.zeros((L, L))
camera = Camera(fig)
# 0 Copera 1 Deserta 2 CorruPto
params[0] = r
params[1] = G
params[2] = c
params[3] = sigma
# ************************************** início*************************************************
inicia_vizinhos(viz)
# todo fazer a amostragem virar uma função que altera o vetor estrat, utilizar multiprocessing nela
for amo in range(0, amostras):  # loop de amostras
    # Monte-Carlo para uma amostra com "total_passos" passos de tempo
    monte_carlo(amo, viz, params, estrategia, estrat_t)
estrat_medio_t = np.sum(estrat_t, axis=1) / amostras
plota_dados(estrat_medio_t)
imprime_dados(estrat_medio_t)
# Geração de figuras
# plt.colorbar()
animation = camera.animate()
animation.save('egalitanima.gif', writer='imagemagick')
