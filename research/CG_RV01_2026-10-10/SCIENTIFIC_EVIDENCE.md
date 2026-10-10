# CG-RV01 — Evidencias cientificas e interpretacao calibrada

**Data da consolidacao:** 10/10/2026. **Fonte primaria da rodada:** \`CG_RV01_PACOTE_GPT.zip\` (hash em PROVENANCE.md). Uma auditoria anterior verificou hashes e recalculou 240 saidas E5; este registro documental nao executou novamente as producoes.

## Mecanismo fisico (DERIVACAO / CODIGO CONSULTADO)
Rede quadrada periodica; vizinhanca de quatro primeiros vizinhos; cinco grupos sobrepostos por agente; G=5, estrategias C, D e P. P significa agente corrupto redistributivo, **nao** punidor. Cada agente P contribui c mas extrai sigma c antes da multiplicacao. Escrevendo nC+nD+nP=G:

\`\`\`text
S = c[nC + (1-sigma)nP]        recurso coletivo liquido
E = sigma c nP                 extracao
B = r S/G                      retorno publico por membro
nC>0:
    piD = B
    piC = B-c+alpha E/nC
    piP = B-c+(1-alpha)sigma c
nC=0:
    piP = B-c+sigma c          retencao integral
\`\`\`

Cada payoff do individuo agrega cinco grupos. A identidade por grupo e \`sum_i pi_i=(r-1)c[nC+(1-sigma)nP]\`. Assim o payoff social medio da rede (cinco jogos por agente) e \`Pi_social=5(r-1)c[rhoC+(1-sigma)rhoP]\`. Com mesma composicao, \`r>1\` e \`sigma>0\`, extrair reduz a producao em comparacao a \`sigma=0\`, mas a comparacao apos evolucao inclui mudancas de \`rho\`.

Para um grupo com nC>0: \`piP-piD=c[(1-alpha)sigma-1]\`, \`piC-piD=c[alpha sigma nP/nC-1]\`, \`piP-piC=sigma c[(1-alpha)-alpha nP/nC]\`. Se nC=0: \`piP-piD=c(sigma-1)\`. **Estas relacoes entre participantes do mesmo grupo nao predizem dominancia espacial global**, pois as vizinhancas sobrepostas de sitios diferentes nao sao identicas. Quando \`sigma>1\`, P pode ter contribuicao publica liquida negativa; interpretacao economica deve ser declarada.

## Well-mixed de G finito (DERIVACAO)
Focal condicionado, x=rhoC, y=rhoD, z=rhoP, q=x+(1-sigma)z,
\`h_G(x)=[1-(1-x)^(G-1)]/x\`, extensao continua \`h_G(0)=G-1\`:

\`\`\`text
F_D = rc(G-1)q/G
F_C = rc[1+(G-1)q]/G - c + alpha sigma c z h_G(x)
F_P = rc[(1-sigma)+(G-1)q]/G - c
      + sigma c[1-alpha+alpha (1-x)^(G-1)]
\`\`\`

Logo \`F_C-F_D=rc/G-c+alpha sigma c z h_G(x)\`: em geral **r nao cancela**. A aproximacao homogenea simplificada e outro objeto. No ponto A foi reportado um foco interior estavel \`(x,y,z)=(0.0102764,0.7841962,0.2055274)\`, autovalores \`-0.00262489 +/- 0.06675161 i\` por grupo. A rede apresentou C medio consideravelmente maior (~0.207, dependente de amostra/protocolo). O foco amortecido nao demonstra ciclo macroscopico. A derivacao finite-G ja esta no apendice da copia de manuscrito recebida: nao reescrever do zero.

## E1 / motor e observaveis (TESTE REPORTADO E REVISADO)
189 casos de contabilidade de grupos e 9.261 comparacoes de payoff registrados no pacote. Auditoria independente executou sete checks em \`src/safety_gate.py\` (PASS em Linux) e reanalisou integridade dos dados E5. Seed interna Numba ja presente no codigo do pacote local, ao contrario do ZIP legado de maio. P01 modifica media condicionada de estrategia ausente de zero para \`NaN\`; preservou dinamica/payoffs presentes bit a bit nos testes declarados; cuidado com leitores de NPZ historicos.

## E3 — baseline C/D (SIMULACAO FINITA REPORTADA)
G=5, K=0.1, c=1, L=200: no ponto r=3.575, 0/30 realizacoes com cooperacao sobrevivente em 12.000 MCS; em r=3.6, 30/30 em 120.000 MCS. **Os horizontes nao sao pareados:** nao chamar o intervalo de estimativa termodinamica precisa de r_crit nem importar diretamente limiar da literatura com K=0.5.

## E5 — mecanismos locais sob intervencoes definidas (REANALISE_VERIFICADA)
Ponto A: \`r=3.3, sigma=1.05, alpha=0.40, K=0.1, G=5, c=1\`, inicializacao aleatoria C/D/P; 6.000 MCS; medias na metade final; 30 seeds/controle/tamanho. Arquivo [E5_ABLATIONS.csv](E5_ABLATIONS.csv) contem resultados de 8 celulas (2 L x 4 variantes). FULL: coexistencia final 29/30 em L100, 30/30 em L200; payoff social medio \`2.140176 +/- 0.094022\` e \`2.226182 +/- 0.003996\`, respectivamente. NO_RETURN: all-P, payoff \`-0.575\`. NO_EXTRACTION e NO_C_FALLBACK_BONUS: all-D, payoff zero. A reanalise conferiu 240 NPZ, hashes, somas de densidades, identidade do ledger e payoff social ponderado; nao refez os 240 Monte Carlo.

**Limite causal:** \`NO_RETURN\` fixa alpha=0 e tambem aumenta a retencao P; \`NO_EXTRACTION\` fixa sigma=0, igualando C e P em payoff; \`NO_C_FALLBACK_BONUS\` retira retorno sem receptor e faz \`piP-piD\` em grupo nC=0 mudar de \`+0.05c\` (FULL) para \`-0.37c\`. Assim nao se demonstrou necessidade universal de uma unica transferencia. Mesmo seed label nao assegura trajetoria acoplada apos divergencia. \`0/30\` nao e probabilidade exatamente zero.

## H4 — achado negativo sobre payoff social (REANALISE_VERIFICADA)
Em FULL, a populacao evoluida apresentou payoff social positivo e maior que NO_EXTRACTION, que terminou em D puro/payoff zero. A identidade para a comparacao contra NO_EXTRACTION e:
\`\`\`text
DeltaPi = 5(r-1)c{ [(rhoC+rhoP)_FULL-(rhoC+rhoP)_NO_EXTRACTION]
                 - sigma rhoP_FULL }
\`\`\`
L100: \`+5.809699\` (composicao) \`-3.669523\` (extracao) \`=+2.140176\` (liquido).
L200: \`+5.568221\` (composicao) \`-3.342039\` (extracao) \`=+2.226182\`.
Logo **a hipotese de piora necessaria entre populacoes evoluidas falhou neste contrafactual**; a perda produtiva a composicao fixa segue exata. Nao e beneficio geral/normativo nem prova de superioridade a outras politicas.

## G4 — oscilacoes, atividade local e escala (SIMULACAO FINITA + ANALISE)
30 trajetorias, 10 por L=100/200/400, ate 24.000 MCS. Quatro janelas, dois tamanhos de blocos e tres bandas espectrais foram aplicados aos **mesmos dados**, sem gerar 24 experiencias independentes. Janela tardia primaria: amplitudes globais \`0.1711, 0.0819, 0.0402\`; locais \`0.3617,0.3669,0.3640\`. Coerencia filtrada \`0.39330,0.18333,0.09221\`. Compatibilidade descritiva com \`A_global~1/L\`, sem ajuste termodinamico confiavel. Interpretacao provisoria: atividade local persistente com reducao de sincronismo global. Nao foi provada existencia de ciclos autonomos, independencia dos dominios, limite L->infinito ou expoente universal.

## Originalidade e integridade epistemica
Reward, redistribuicao, corrupcao e ciclos espaciais ja possuem precedentes. A novidade e hipotese estreita: acoplamento da extracao *antes da amplificacao*, retencao privada e transferencia seletiva para C na geometria espacial dada, mais consequencias em interfaces e produtividade evoluida. Nao houve releitura integral da literatura nesta rodada documental. Oito ZIPs complementares de dados e 18 linhagens de paineis nao estavam integralmente disponiveis; ver [PROVENANCE.md](PROVENANCE.md).
