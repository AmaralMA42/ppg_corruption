# Documento Ontologico Central - Corruption Games

## Core Identity

Projeto de pesquisa em teoria dos jogos evolucionaria espacial inspirado em clientelismo politico, coronelismo e estruturas de dependencia social observadas em sistemas politicos reais.

O objetivo central e investigar como corrupcao redistributiva local pode sustentar cooperacao enquanto degrada produtividade coletiva e gera estruturas clientelistas emergentes.

O projeto combina:

- teoria dos jogos evolucionaria;
- fisica estatistica;
- criticalidade;
- sistemas complexos;
- dinamica espacial;
- cooperacao;
- corrupcao sistemica.

## Estrategias

### C - Cooperators

Agentes que contribuem para o bem publico pagando custo `c`.

### D - Defectors

Agentes que nao contribuem e apenas exploram o sistema coletivo.

### P - Corrupt Agents

Agentes corruptos que:

- extraem parte do recurso coletivo antes da multiplicacao do public goods;
- redistribuem localmente parte do recurso roubado para cooperadores vizinhos;
- tentam estabilizar sua sobrevivencia criando dependencia local, isto e, um "curral eleitoral".

`P` nao representa punicao, instituicoes ou fiscalizacao.

`P` representa corrupcao clientelista redistributiva.

## Mecanica Central

A corrupcao ocorre antes da multiplicacao do bem publico:

```text
L = r c (C + P - sigma P) / G
```

onde:

- `sigma` controla a intensidade da corrupcao/expropriacao;
- corruptos reduzem a produtividade sistemica;
- maior corrupcao reduz a eficiencia coletiva.

## Redistribuicao Clientelista

Corruptos redistribuem uma fracao `alpha` do recurso roubado para cooperadores vizinhos:

```text
alpha sigma c P / C
```

Interpretacao:

- propina local;
- compra de apoio politico;
- manutencao de dependencia social;
- estabilizacao espacial da corrupcao.

`alpha` controla a intensidade do clientelismo.

Valores muito altos, especialmente `alpha > 0.5`, possuem interpretacao economica limitada, mas podem gerar regimes fisicos interessantes.

## Hipotese Central

Corrupcao redistributiva local pode:

- impedir colapso completo da cooperacao;
- estabilizar coexistencia espacial;
- gerar ciclos evolutivos;
- produzir estados metastaveis;
- criar estruturas clientelistas auto-organizadas;
- sustentar cooperacao toxica de baixa produtividade.

## Motivacao Social

Inspirado em:

- coronelismo brasileiro;
- clientelismo politico;
- "rouba mas faz";
- compra local de apoio politico;
- dependencia economica regional.

O modelo busca formalizar matematicamente mecanismos sociais de manutencao da corrupcao atraves de redistribuicao seletiva local.

## Motivacao Fisica

Investigar:

- coexistencia espacial;
- criticalidade;
- transicoes de fase;
- sincronizacao;
- ciclos evolutivos;
- metastabilidade;
- auto-organizacao;
- competicao entre exploracao e sustentacao cooperativa.

## Dinamica

Atualizacao por dinamica de Fermi:

```text
W = 1 / (1 + exp(-DeltaPi / K))
```

com:

- rede quadrada periodica inicialmente;
- possibilidade futura de redes triangulares e outras topologias;
- interacao local via public goods game espacial.

## Limite Fisico Importante

Na ausencia de corrupcao, o modelo deve recuperar o spatial public goods game classico:

- mesmos regimes fisicos;
- mesmos limiares criticos;
- mesma transicao de fase conhecida da literatura.

Esse e um criterio fundamental de validacao.

## Objetivos Cientificos

### Curto prazo

- validar implementacao atual;
- reproduzir PGG classico;
- mapear diagramas de fase;
- detectar coexistencia e estados absorventes.

### Medio prazo

- estudar criticalidade;
- investigar ciclos espaciais;
- analisar sincronizacao temporal;
- medir flutuacoes e variancias.

### Longo prazo

- finite-size scaling;
- classes de universalidade;
- criticalidade auto-organizada;
- dinamica em redes complexas;
- publicacao em PRE, Chaos ou PRL.

## Observaveis Atuais

- fracao media das estrategias;
- payoff medio por estrategia;
- variancia temporal;
- atividade normalizada por MCS;
- tempo de absorcao por amostra;
- autocorrelacao temporal;
- FFT/espectro temporal;
- periodo dominante;
- concentracao espectral do pico dominante;
- trajetorias temporais;
- diagramas de fase para fracoes, variancia temporal e metricas espectrais.

## Observaveis Futuros

- susceptibility;
- Binder cumulant;
- distribuicao de clusters;
- lifetime de dominios;
- avalanche statistics;
- correlacao espacial;
- domain wall dynamics;
- synchronization order parameters.

## Filosofia Computacional

Prioridade:

- fidelidade fisica do modelo;
- clareza cientifica;
- modularidade suficiente para expansao.

O projeto nao busca arquitetura enterprise. Busca infraestrutura cientifica modular e reproduzivel.

Ferramentas principais:

- Python;
- NumPy;
- Numba;
- SciPy;
- Matplotlib.

Possivel expansao:

- NetworkX;
- Polars;
- JAX;
- PyTorch;
- Dask.

## Estrutura Computacional Planejada

Separacao futura entre:

- modelos;
- dinamica;
- topologias;
- observaveis;
- experimentos;
- visualizacao.

Objetivo:

- permitir rapida exploracao fisica;
- facilitar sweeps e phase diagrams;
- manter simplicidade conceitual.

## Referencias Conceituais

Inspirado principalmente por:

- Matjaz Perc;
- Gyorgy Szabo;
- Christoph Hauert;
- Jeferson Arenzon;
- Jafferson Kamphorst;
- literatura classica de EGT espacial e fisica estatistica.

## Assinatura Conceitual do Projeto

Corrupcao nao atua apenas como destruicao sistemica.

Ela pode:

- sustentar cooperacao local;
- criar dependencias toxicas;
- estabilizar ecossistemas sociais de baixa produtividade;
- produzir estruturas espaciais persistentes;
- gerar dinamica coletiva emergente complexa.
