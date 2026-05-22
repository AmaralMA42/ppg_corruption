# Corruption Games

Simulacao computacional de um jogo dos bens publicos espacial com corrupcao clientelista redistributiva. O projeto investiga como agentes corruptos podem degradar a produtividade coletiva e, ao mesmo tempo, sustentar cooperacao local por meio de redistribuicao seletiva, produzindo coexistencia, ciclos, metastabilidade e estruturas espaciais persistentes.

Este repositorio faz parte de uma pesquisa em teoria dos jogos evolucionaria, fisica estatistica e sistemas complexos. A ontologia conceitual completa esta em [`DOCUMENTO_ONTOLOGICO.md`](DOCUMENTO_ONTOLOGICO.md).

## Ideia Central

O modelo possui tres estrategias:

- `C` - cooperadores: contribuem para o bem publico pagando custo `c`.
- `D` - defectores: nao contribuem e exploram o bem coletivo.
- `P` - corruptos: extraem parte do recurso coletivo antes da multiplicacao e redistribuem localmente parte do recurso roubado para cooperadores vizinhos.

Importante: `P` significa **corrupt agents**. `P` nao representa punidores, fiscalizacao, sancao institucional ou mecanismo de enforcement. A interpretacao correta e corrupcao clientelista redistributiva.

## Modelo

A simulacao implementa um spatial public goods game em rede quadrada periodica. Cada jogador interage localmente com sua vizinhanca e atualiza sua estrategia por imitacao estocastica via regra de Fermi.

O beneficio local do bem publico e calculado como:

```text
L = r c (C + P - sigma P) / G
```

onde:

- `r` e o fator de multiplicacao do bem publico;
- `c` e o custo de contribuicao;
- `G` e o tamanho do grupo local;
- `sigma` controla a intensidade da corrupcao/expropriacao;
- `P` reduz a produtividade sistemica antes da multiplicacao.

A redistribuicao clientelista e controlada por `alpha`, que define a fracao do recurso roubado redistribuida localmente para cooperadores. Valores altos de `alpha` podem ser fisicamente interessantes, embora tenham interpretacao economica mais limitada.

## Hipotese De Pesquisa

Corrupcao redistributiva local pode impedir o colapso completo da cooperacao, mas ao custo de criar uma cooperacao toxica, de baixa produtividade, baseada em dependencia local. O interesse fisico esta em observar coexistencia espacial, ciclos evolutivos, estados absorventes, transicoes de fase, flutuacoes e possiveis regimes criticos.

## Implementacao Atual

A versao correta e mais recente do codigo e a versao modular em [`src`](src).

```text
src/
  config.py           parametros centrais da simulacao
  core_simulation.py  dinamica, payoff, vizinhanca e Monte Carlo
  run_sampling.py     simulacoes multi-amostra
  run_sweep.py        varreduras 1D em r, sigma e alpha
  run_phaseD.py       diagramas de fase 2D
  run_visual.py       visualizacoes espaciais e GIFs
  plotting.py         rotinas de plotagem
  utils.py            utilitarios de dados
```

Arquivos monoliticos antigos e pastas de experimentos anteriores permanecem no repositorio como material legado ou historico experimental, mas nao devem ser tratados como a implementacao principal.

## Dependencias

Dependencias principais:

- Python 3.10+
- NumPy
- Numba
- SciPy
- Matplotlib
- Pillow

Dependencias auxiliares usadas por scripts de debug/legado:

- psutil
- celluloid

Instalacao sugerida:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como Rodar

Entre na pasta do projeto:

```bash
cd ppg_corruption
```

Rodar uma simulacao multi-amostra:

```bash
python src/run_sampling.py
```

Rodar varreduras 1D em parametros como `r`, `sigma` e `alpha`:

```bash
python src/run_sweep.py
```

Rodar diagramas de fase 2D:

```bash
python src/run_phaseD.py
```

Gerar visualizacoes espaciais e animacoes:

```bash
python src/run_visual.py
```

Os parametros principais ficam em [`src/config.py`](src/config.py), na classe `SimulationConfig`.

## Parametros Principais

- `L`: tamanho linear da rede quadrada.
- `amostras`: numero de amostras independentes.
- `total_passos`: numero de passos de Monte Carlo.
- `percent_avg_MC`: fracao final usada para media termica.
- `seed`: semente aleatoria.
- `k`: ruido/irracionalidade da regra de Fermi.
- `r`: fator de multiplicacao do bem publico.
- `G`: tamanho do grupo local.
- `c`: custo de contribuicao.
- `sigma`: intensidade da corrupcao/expropriacao.
- `alpha`: intensidade da redistribuicao clientelista.
- `cond_ini`: condicao inicial das estrategias.

## Saidas

As simulacoes podem produzir:

- trajetorias temporais das fracoes `rho_C`, `rho_D` e `rho_P`;
- payoff medio por estrategia;
- medias em estado estacionario apos termalizacao;
- variancias temporais;
- arquivos `.dat` e `.npy` em `data/`;
- figuras, mapas de fase e animacoes em `figures/` ou no diretorio de execucao, dependendo do script.

## Observaveis Atuais

- fracao media das estrategias;
- payoff medio por estrategia;
- variancia temporal;
- trajetorias temporais;
- mapas 1D e 2D em parametros de controle.

## Objetivos

Curto prazo:

- validar a implementacao modular;
- recuperar o PGG espacial classico no limite sem corrupcao;
- mapear diagramas de fase;
- detectar coexistencia e estados absorventes.

Medio prazo:

- estudar criticalidade;
- investigar ciclos espaciais;
- analisar sincronizacao temporal;
- medir flutuacoes, variancias e indicadores dinamicos.

Longo prazo:

- finite-size scaling;
- classes de universalidade;
- criticalidade auto-organizada;
- dinamica em redes complexas;
- publicacao cientifica.

## Validacao Fisica

Um criterio essencial de validacao e que, na ausencia de corrupcao, o modelo recupere o spatial public goods game classico: regimes conhecidos, limiares criticos e transicoes de fase esperadas na literatura.

## Status

Projeto em desenvolvimento ativo. A prioridade atual e fidelidade fisica, clareza cientifica e modularidade suficiente para explorar rapidamente regimes, sweeps e diagramas de fase.
