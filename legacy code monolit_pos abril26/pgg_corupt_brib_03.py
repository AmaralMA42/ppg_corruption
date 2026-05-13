import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import time
from pathlib import Path
from numba import jit, prange
from celluloid import Camera
from config import SimulationConfig
 # todo !!!!!salvar payoff medio e payoff do sitio evoluindo para verificar se na oexplode


cfg = SimulationConfig()
L = cfg.L
amostras = cfg.amostras
total_passos = cfg.total_passos
passos_media = cfg.passos_media
seed = cfg.seed
k = cfg.k
r = cfg.r
G = cfg.G
c = cfg.c
sigma = cfg.sigma
alpha = cfg.alpha
creat_snapshot = cfg.create_snapshot
deldata = cfg.deldata
framerate = cfg.framerate
fpsgif = cfg.fpsgif
passo_filma_inicio = cfg.passo_filma_inicio
cond_ini = cfg.cond_ini


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
np.random.seed(seed)
start_time = time.time()



#Gerador da imagem
def generate_imagegrid(imagegrid, estrategia, L):
    for i in range(0, L):
        for j in range(0, L):
                imagegrid[i][j] = estrategia[i * L + j]
                # 0=C,1=D,2=P


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


@jit(nopython=True, fastmath=True)  # todo Gasto de tempo enorme, como otimizar?
def calc_fracs2(amo, estrat_t, estrategia, passo_atual, total_jog):
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

@jit(nopython=True, fastmath=True,  parallel=False)  # todo olhar!!!
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


        pay_atual = payoff[atual]
        pay_viz = payoff[vizsorteado]

        var_pay = pay_viz - pay_atual

        prob =  np.random.random() # list_prob[cont]                          # list_prob[cont] # random.random()  # probabilidade aleatória
        chance_muda = prob_flip(var_pay, k)  # Probabilidade de fermi

        if prob < chance_muda:

            estrategia[atual] = estrategia[vizsorteado]  # mudança da estratégia do sítio central
            atualiza_payoff_local_extra(atual, estrategia, payoff, viz, params)

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

#@jit(nopython=True, fastmath=True,  parallel=False)
def monte_carlo(amo, viz, params, estrategia, payoff, estrat_t, payavg_t, imagegrid, camera, total_jog, total_passos, framerate, passo_filma_inicio, L):
    # Inicio das estratégias para amostra
    inicia_estrategias(estrategia, total_jog, L, cond_ini)
    # inicia os payoffs de cada sítio
    for i in range(total_jog):
        payoff[i] = calcula_payoff(i, estrategia, viz, params)

    # Começo da simulação de Monte-Carlo
    for passo_atual in range(0, total_passos):
        # calcular todas estatísticas da população no passo atual
        cfrac, dfrac, pfrac = calc_fracs(estrategia, total_jog)
        estrat_t[0, amo, passo_atual] = cfrac
        estrat_t[1, amo, passo_atual] = dfrac
        estrat_t[2, amo, passo_atual] = pfrac

        pay_strat = calc_payoff_por_estrategia(estrategia, payoff, total_jog)
        payavg_t[0, amo, passo_atual] = pay_strat[0]  # C
        payavg_t[1, amo, passo_atual] = pay_strat[1]  # D
        payavg_t[2, amo, passo_atual] = pay_strat[2]  # P

#        payavg_t[amo, passo_atual] = mean_payoff(payoff)
        # payavg_t = payavg_t + 0
        # Gerando as imagens da rede

        if creat_snapshot:
            if passo_atual % framerate == 0 and passo_atual >= passo_filma_inicio:
                generate_imagegrid(imagegrid, estrategia, L)
                plt.figure(amo)
                plt.imshow(imagegrid, vmin=0, vmax=2, cmap='brg', interpolation='none')
                plt.title(f"mcs={passo_atual} | r={r}, σ={sigma}, α={alpha}, k={k}")
                plt.savefig(FIGURES_DIR / f"{passo_atual}.png")
                camera.snap()

        # Etapa de atualização da estratégia
        atualiza_total_estrat(estrategia, payoff, viz, params, total_jog)  # atualiza cada rede individualmente

    if creat_snapshot:
        plt.figure(amo)
        plt.colorbar(ticks=[0, 1, 2, 3])


# @jit(nopython=True, fastmath=True,  parallel=False) Numba não compatível com impressão aparentemente
def imprime_dados(estrat_medio_t, total_passos, start_time):
    arquivo2 = open(DATA_DIR / f"subpop_r{r}_sigma{sigma}.dat", "w+")
    for cont in range(0, total_passos):
        arquivo2.write(f"{cont} {estrat_medio_t[0, cont]:.4f} {estrat_medio_t[1, cont]:.4f} "
                       f"{estrat_medio_t[2, cont]:.4f} \n")
    print(f"----- {(time.time() - start_time):.4f} seconds or {((time.time() - start_time) / 60.0):.4f} minutes-----")
    arquivo2.close()


def plota_dados(estrat_medio_t):
    plt.figure()
    plt.title('Evolução média das subpopulações')
    plt.xlabel('Número de Passos')
    plt.ylabel(r'$<\rho >$')
    plt.plot(estrat_medio_t[0, :], label='C')
    plt.plot(estrat_medio_t[1, :], label='D')
    plt.plot(estrat_medio_t[2, :], label='P')
    plt.ylim(0, 1)
    # salvar parametros nos graficos
    param_text = (
        f"L={L}\n"
        f"r={r}\n"
        f"$\\sigma$={sigma}\n"
        f"$\\alpha$={alpha}\n"
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

def plota_todas_amostras(estrat_t, estrat_medio_t):
    plt.figure()

    # trajetórias individuais
    for i in range(estrat_t.shape[1]):
        plt.plot(estrat_t[0, i, :], alpha=0.2, linestyle='-')
        plt.plot(estrat_t[1, i, :], alpha=0.2, linestyle='--')
        plt.plot(estrat_t[2, i, :], alpha=0.2, linestyle=':')

    # média (destacada)
    plt.plot(estrat_medio_t[0, :], label='C (média)', linewidth=2., color='blue')
    plt.plot(estrat_medio_t[1, :], label='D (média)', linewidth=2., color='red')
    plt.plot(estrat_medio_t[2, :], label='P (média)', linewidth=2., color='green')

    plt.ylim(0, 1)
    plt.xlabel('Número de Passos')
    plt.ylabel(r'$\langle \rho \rangle$')

    # parâmetros discretos
    param_text = (
        f"L={L}\n"
        f"r={r}\n"
        f"$\\sigma$={sigma}\n"
        f"$\\alpha$={alpha}\n"
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

def plota_payoff_por_estrategia(payavg_t, payavg_medio_t):
    plt.figure()

    # trajetórias individuais
    for i in range(payavg_t.shape[1]):
        plt.plot(payavg_t[0, i, :], alpha=0.2, linestyle='-')
        plt.plot(payavg_t[1, i, :], alpha=0.2, linestyle='--')
        plt.plot(payavg_t[2, i, :], alpha=0.2, linestyle=':')

    # média (destacada)
    plt.plot(payavg_medio_t[0, :], label='C payoff', linewidth=2.)
    plt.plot(payavg_medio_t[1, :], label='D payoff', linewidth=2.)
    plt.plot(payavg_medio_t[2, :], label='P payoff', linewidth=2.)

    plt.xlabel('Número de Passos')
    plt.ylabel(r'$\langle \Pi \rangle$')

    param_text = (
        f"L={L}\n"
        f"r={r}\n"
        f"$\\sigma$={sigma}\n"
        f"$\\alpha$={alpha}\n"
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
    total_jog=L * L
    estrategia = np.zeros(total_jog, dtype=int)
    payoff = np.zeros(total_jog)
    estrat_t = np.zeros((3, amostras, total_passos))
    payavg_t = np.zeros((3, amostras, total_passos))
    # Definição de vizinhos
    viz = np.zeros((total_jog, 4), dtype=int)  # matriz contendo os vizinhos 0=cima,1=direita,2=baixo,3=esquerda
    #Variáveis da imagem
    fig = plt.figure()
    imagegrid = np.zeros((L, L))
    camera = Camera(fig)
    # 0 Copera 1 Deserta 2 CorruPto
    params = cfg.simulation_params()
#    params[0] = r
#    params[1] = G
#    params[2] = c
#    params[3] = sigma
#    params[4] = k
#    params[5] = alpha
    # ************************************** início*************************************************
    inicia_vizinhos(viz, total_jog, L)
    # todo fazer a amostragem virar uma função que altera o vetor estrat, utilizar multiprocessing nela
    for amo in range(0, amostras):  # loop de amostras
        # Monte-Carlo para uma amostra com "total_passos" passos de tempo
        estrategia[:] = 0
        payoff[:] = 0
        monte_carlo(amo, viz, params, estrategia, payoff, estrat_t,payavg_t, imagegrid, camera, total_jog, total_passos, framerate, passo_filma_inicio, L)
    estrat_medio_t = np.sum(estrat_t, axis=1) / amostras
    payavg_medio_t = np.sum(payavg_t, axis=1) / amostras
#    for i in range(amostras):
#        plota_dados(estrat_t[:,i,:])
#    plota_dados(estrat_medio_t)
    plota_todas_amostras(estrat_t, estrat_medio_t)
    plota_payoff_por_estrategia(payavg_t, payavg_medio_t)
    imprime_dados(estrat_medio_t, total_passos, start_time)
    #Geração de figuras
    #plt.colorbar()
    if creat_snapshot:
        animation = camera.animate()
        animation.save(FIGURES_DIR / "egalitanima.gif", writer=PillowWriter(fps=fpsgif))


if __name__ == "__main__":
    main()
