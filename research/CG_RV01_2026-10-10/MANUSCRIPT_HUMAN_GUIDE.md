# CG-RV01 — Orientacoes para revisao humana do artigo (NAO EDITADO)

Esta pagina NAO e uma revisao automatica aplicada; o autor decidira se, quando e como incorporar os pontos. O manuscrito consultado foi a copia `project/corruption_games_PRE_enriched.tex` contida em `CG_RV01_PACOTE_GPT.zip`. Nao presumir equivalencia com uma versao posterior do Work.

## Alteracoes conceituais prioritarias
1. **Estrutura matematica:** manter o finite-G well-mixed ja presente no apendice; contrastar explicitamente foco espiral amortecido desse baseline com flutuacoes espaciais. Evitar chamada de 'first-neighbor correlations' se nenhuma correlacao de vizinhanca e calculada na secao.
2. **Welfare:** separar perda contabilidade a composicao FIXA (identidade exata) de payoff social em populacoes EVOLUIDAS (resultado dependente de composicao, parametro e comparador). Preservar o resultado E5 FULL > NO_EXTRACTION no ponto A, sem enunciar beneficio universal da corrupcao.
3. **Causalidade:** a ablacao alpha=0 muda transferencias a C E retencao privada de P; o controle de fallback muda vantagem P/D para nC=0; sigma=0 iguala C e P. Mostrar essas intervencoes pelo nome operacional e nao prometer isolamento de mecanismo unico.
4. **Oscilacoes:** descrever evidencia de atividade local em varios L e perda da componente global; nao alegar ciclos macroscópicos persistentes, expoentes universais, transicao de fase, sincronizacao completa ou dominio infinito.
5. **Fenomenologia:** preservar discussao fisica sobre interfaces, redistribuicao, invasoes e organizacao espacial. Nao transformar todo o artigo em relatorio de QA; metodologia de reparos, verificacoes de seeding e tabelas de hashes podem ir ao suplementar.
6. **Semantica:** explicitar sigma>1 (P pode reduzir recurso publico liquido abaixo de zero), regra sem C e interpretacao social que nao substitui dinamica matematica. P nao e punidor.

## Defeitos materiais a conferir na copia vigente
- `eq:oscillatory_point` referencia interna deve ser definida e possivelmente renomeada para `eq:anchor_point`, pois o ponto e ilustrativo e nao um oscilador termodinamico certificado.
- Paragrafo duplicado na secao de payoffs; eliminar duplicacao apenas apos conferir a versao atual.
- Rotas `nfigs/` e selecao antiga `snpa1.png` / `snap1.png`; nao substituir a procedencia por semelhanca visual.
- Figuras de tamanhos historicos com alpha=0.35 nao devem ser apresentadas como comparacao pareada da ancora alpha=0.40.
- Lacunas de autoria, metadados, financiamento, disponibilidade de codigo/dados e 18 linhagens de painéis antigas.

## O que NAO reescrever automaticamente
- Nao remover heatmaps e sweeps so porque a procedencia esta aberta; manter como exploratorios ate haver verificacao, ou separar claramente painéis nao certificados.
- Nao alegar que a intuicao do projeto ('cooperacao toxica') e refutada em geral. O que foi contrariado e a afirmacao *universal* de welfare evoluido inferior em um comparador especifico.
- Nao duplicar as equacoes finite-G que ja constam do apendice.
- Nao promover figura/claim apenas porque o patch editorial compila.

## Gate editorial
G0 aberto (fontes de figura), G4 provisorio (macroscopico), G5 dependente de ratificacao humana. O autor deve comparar o `.tex` local corrente com o SHA registrado em PROVENANCE.md antes de editar.
