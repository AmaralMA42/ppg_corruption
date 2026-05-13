import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import time
import random
from numba import jit, prange
from celluloid import Camera
import os

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
# Parâmetros da simulação:
L = 200  # tamanho da rede
amostras = 1
total_passos = 500
passos_media = int(0.9 * total_passos)
framerate = 50
fpsgif=5
passo_filma_inicio = 0
k = 0.1     # Irracionalidade
r = 4.0  # Multiplication factor
G = 5  # Group Size
c = 1
sigma = 1.2
total_jog = L * L
start_time = time.time()
np.random.seed()

#Gerador da imagem
def generate_imagegrid(imagegrid, estrategia, L):
    for i in range(0, L):
        for j in range(0, L):
                imagegrid[i][j] = estrategia[i * L + j]
                # 0=C,1=D,2=P


@jit(nopython=True, fastmath=True, parallel=False)
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
def inicia_estrategias(estrategia, total_jog):
    # Inicialização das estratégias
    for jogador_atual in range(total_jog):  # Lembete, for in range vai até total_jog-1!!
        estrategia[jogador_atual] = np.random.randint(3)  # Estado inicial do jogador como aleatório

#Versão MAIS eficiente (nível ideal)
#Atualizar contadores durante a simulação
# mantém Nc, Nd, Np globais e atualiza quando um jogador muda estratégia
@jit(nopython=True, fastmath=True)  # todo Gasto de tempo enorme, como otimizar?
def calc_fracs(amo, estrat_t, estrategia, passo_atual, total_jog):
    c = 0
    d = 0
    p = 0

    for i in range(total_jog):
        if estrategia[i] == 0:
            c += 1
        elif estrategia[i] == 1:
            d += 1
        else:
            p += 1

    estrat_t[0, amo, passo_atual] = c / total_jog
    estrat_t[1, amo, passo_atual] = d / total_jog
    estrat_t[2, amo, passo_atual] = p / total_jog
#    estrat_t[0, amo, passo_atual] = np.sum(estrategia == 0) / total_jog  # C
#    estrat_t[1, amo, passo_atual] = np.sum(estrategia == 1) / total_jog  # D
#    estrat_t[2, amo, passo_atual] = np.sum(estrategia == 2) / total_jog  # P


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
    r, G, c, sigma, _ = params
    return float(r * c * (C + P - sigma * P) / G)

@jit(nopython=True, fastmath=True,  parallel=False)
def conta_estrat(sitio, viz, estrategia):
    vec_estrat = np.zeros(3)  # (C,D,P)
    for j in range(4):  # Conta quantas estratégias existem no grupo centrado no sítio
        vec_estrat[estrategia[viz[sitio, j]]] += 1
    vec_estrat[estrategia[sitio]] += 1
    return vec_estrat

@jit(nopython=True, fastmath=True,  parallel=False) # OTIMIZAR!!!!!!
def calcula_payoff(sitio, estrategia, viz, params):
    c = params[2]
    sigma = params[3]
    payoff_sitio = 0
    Nc , Nd, Np = conta_estrat(sitio, viz, estrategia)
    payoff_sitio += public_good_benefit(Nc,Nd,Np, params) # matriz_pay_[int(Nc),int(Nd)] # r * (Nc + Np - sigma * Np) / G
# jogo centrado em cada um dos 4 vizinhos
    for i in range(4):
        sitio_temp = viz[sitio, i]
        Nc, Nd, Np = conta_estrat(sitio_temp, viz, estrategia)
        payoff_sitio += public_good_benefit(Nc,Nd,Np, params) # matriz_pay_[int(Nc),int(Nd)] # r * (Nc + Np - sigma * Np) / G
    if estrategia[sitio] == 0:
        payoff_sitio = payoff_sitio - 5 * c
    if estrategia[sitio] == 2:
        payoff_sitio = payoff_sitio - 5 * c + 5 * sigma * c
    return payoff_sitio


@jit(nopython=True, fastmath=True,  parallel=False)  # todo olhar!!!
def atualiza_total_estrat(estrategia, viz, params, total_jog):
    k = params[4]
#    list_atual = np.random.randint(0, total_jog, size=total_jog)  # lista de jogadores aleatórios para um  MCS
#    list_sorteio = np.random.randint(0, 4, size=total_jog)
#    list_prob = np.random.random(size=total_jog)
    for cont in range(0, total_jog):
        atual = np.random.randint(0, total_jog) # list_atual[cont]                       # list_atual[cont]  # np.random.randint(0, total_jog)
        pay_atual = calcula_payoff(atual, estrategia, viz, params)
        vizsorteado = viz[atual, np.random.randint(0, 4)]  # sorteio de um vizinho aleatório de 0 a 3 #random.randint(0, 3)
        pay_viz = calcula_payoff(vizsorteado, estrategia, viz, params)
        var_pay = pay_viz - pay_atual
        prob =  np.random.random() # list_prob[cont]                          # list_prob[cont] # random.random()  # probabilidade aleatória
        chance_muda = prob_flip(var_pay, k)  # Probabilidade de fermi
        if prob < chance_muda:
            estrategia[atual] = estrategia[vizsorteado]  # mudança da estratégia do sítio central


#@jit(nopython=True, fastmath=True,  parallel=False)
def monte_carlo(amo, viz, params, estrategia, estrat_t, imagegrid, camera, total_jog, total_passos, framerate, passo_filma_inicio, L):
    # Inicio das estratégias para amostra
    inicia_estrategias(estrategia, total_jog)
    # Começo da simulação de Monte-Carlo
    for passo_atual in range(0, total_passos):
        # Etapa de atualização da estratégia
        atualiza_total_estrat(estrategia, viz, params, total_jog)  # atualiza cada rede individualmente
        # calcular todas estatísticas da população no passo atual
        calc_fracs(amo, estrat_t, estrategia, passo_atual, total_jog)
        # Gerando as imagens da rede
        if passo_atual % framerate == 0 and passo_atual >= passo_filma_inicio:
            generate_imagegrid(imagegrid, estrategia, L)
            plt.figure(1)
            plt.imshow(imagegrid, vmin=0, vmax=2, cmap='brg', interpolation='none')
            plt.title(f"C=0 D=1 P=2, mcs={passo_atual}")
            plt.savefig('figures/%s.png' % passo_atual)
            camera.snap()
    plt.figure(1)
    plt.colorbar(ticks=[0, 1, 2, 3])

# @jit(nopython=True, fastmath=True,  parallel=False) Numba não compatível com impressão aparentemente
def imprime_dados(estrat_medio_t, total_passos, start_time):
    arquivo2 = open(f"data/subpop_r{r}_sigma{sigma}.dat", "w+")
    for cont in range(0, total_passos):
        arquivo2.write(f"{cont} {estrat_medio_t[0, cont]:.4f} {estrat_medio_t[1, cont]:.4f} "
                       f"{estrat_medio_t[2, cont]:.4f} \n")
    print(f"----- {(time.time() - start_time):.4f} seconds or {((time.time() - start_time) / 60.0):.4f} minutes-----")
    arquivo2.close()


def plota_dados(estrat_medio_t):
    plt.figure(2)
    plt.title('Evolução média das subpopulações')
    plt.xlabel('Número de Passos')
    plt.ylabel(r'$<\rho >$')
    plt.plot(estrat_medio_t[0, :], label='C')
    plt.plot(estrat_medio_t[1, :], label='D')
    plt.plot(estrat_medio_t[2, :], label='P')
    plt.ylim(0, 1)
    param_text = (
        f"L={L}\n"
        f"r={r}\n"
        f"$\\sigma$={sigma}\n"
        f"k={k}"
    )

    plt.text(
        0.98, 0.02, param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    plt.legend()
    plt.show()

def main():
    # Definição de variáveis do jogo
    estrategia = np.zeros(total_jog, dtype=int)
    estrat_t = np.zeros((3, amostras, total_passos))
    params = np.zeros(5)
    # Definição de vizinhos
    viz = np.zeros((total_jog, 4), dtype=int)  # matriz contendo os vizinhos 0=cima,1=direita,2=baixo,3=esquerda
    #Variáveis da imagem
    fig = plt.figure()
    imagegrid = np.zeros((L, L))
    camera = Camera(fig)
    # 0 Copera 1 Deserta 2 CorruPto
    params[0] = r
    params[1] = G
    params[2] = c
    params[3] = sigma
    params[4] = k
    # ************************************** início*************************************************
    inicia_vizinhos(viz, total_jog, L)
    # todo fazer a amostragem virar uma função que altera o vetor estrat, utilizar multiprocessing nela
    for amo in range(0, amostras):  # loop de amostras
        # Monte-Carlo para uma amostra com "total_passos" passos de tempo
        monte_carlo(amo, viz, params, estrategia, estrat_t, imagegrid, camera, total_jog, total_passos, framerate, passo_filma_inicio, L)
    estrat_medio_t = np.sum(estrat_t, axis=1) / amostras
    plota_dados(estrat_medio_t)
    imprime_dados(estrat_medio_t, total_passos, start_time)
    #Geração de figuras
    #plt.colorbar()
    animation = camera.animate()
    animation.save("figures/egalitanima.gif", writer=PillowWriter(fps=fpsgif))


if __name__ == "__main__":
    main()

