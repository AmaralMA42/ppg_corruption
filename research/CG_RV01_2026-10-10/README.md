# CG-RV01 — Registro cientifico de 10/10/2026

**Status: KNOWLEDGE_INTEGRATED / EVIDENCE_GRADED / NON_CANONICAL / NO_CODE_OR_TEX_CHANGES**

Esta pasta preserva o conhecimento produzido pela missao CG-RV01 e pela auditoria independente de 10/10/2026. **Nao e um codigo de producao, nao substitui o manuscrito vigente e nao ratifica hipoteses abertas.** A branch foi criada diretamente de `master` no commit `737a41b63e91c3b32369147f9e25f6d05dae6c97` (25/06/2026) apenas para gravacao documental. Ela pode ser divergente da instancia Work local mais recente.

## Onde comecar
1. [SCIENTIFIC_EVIDENCE.md](SCIENTIFIC_EVIDENCE.md): modelo, derivacoes, resultados numericos sustentados e limites.
2. [E5_ABLATIONS.csv](E5_ABLATIONS.csv): dados agregados do experimento causal E5.
3. [OSCILLATIONS_SCALING.csv](OSCILLATIONS_SCALING.csv): indicadores descritivos de oscilacao para L=100, 200, 400.
4. [HYPOTHESES_AND_GATES.md](HYPOTHESES_AND_GATES.md): o que foi apoiado, refutado ou continua em aberto.
5. [SOFTWARE_BACKLOG.md](SOFTWARE_BACKLOG.md): riscos de engenharia e testes antes de alterar src.
6. [MANUSCRIPT_HUMAN_GUIDE.md](MANUSCRIPT_HUMAN_GUIDE.md): sugestoes editoriais sem modificar .tex.
7. [PROVENANCE.md](PROVENANCE.md): origem, hashes e criterios de nao-superinterpretacao.
8. [FUTURE_EXPERIMENT.md](FUTURE_EXPERIMENT.md): experimento de redistribuicao seletiva planejado, NAO EXECUTADO.

## Resultados centrais preservados
- A contabilidade do PGG com P foi verificada; perda de produtividade por extracao a **composicao fixa** NAO implica perda de payoff na comparacao de **populacoes evoluidas**.
- No ponto A (r=3.3, sigma=1.05, alpha=0.40, K=0.1), as intervencoes definidas de E5 tiveram estados finais distintos em L=100 e 200; os contrastes nao isolam universalmente cada mecanismo.
- O comparador evoluido `NO_EXTRACTION` terminou em D e payoff social zero; o modelo FULL obteve payoff social positivo. A hipotese de perda evoluida necessaria e contrariada neste comparador, nao universalmente refutada.
- L=100, 200, 400 mostram amplitude oscilatoria global decrescente com amplitude local aproximadamente constante; limite termodinamico continua aberto.
- O well-mixed correto de grupos finitos condiciona a estrategia focal e preserva a dependencia em r; a derivacao ja consta do apendice do manuscrito disponibilizado no pacote.
- No pacote local, P01 (estrategia ausente => payoff condicionado NaN) ja estava aplicado e aprovado nos testes de regressao; nao reaplicar sem necessidade.

## Hierarquia de evidencia
**DERIVACAO** = identidade matematica sob as regras; **REANALISE_VERIFICADA** = calculo independente dos dados disponiveis, nao rerun de Monte Carlo; **SIMULACAO_REPORTADA** = experimento do pacote local; **HIPOTESE** = explicacao fisica a testar; **PROPOSTA** = patch ou texto para humano avaliar.

**Nao escrever no .tex, no src nem na branch master por interpretacao automatica destes documentos.** G0 (figuras), G4 (limite oscilatorio) e G5 (aceite cientifico/editorial) continuam abertos.
