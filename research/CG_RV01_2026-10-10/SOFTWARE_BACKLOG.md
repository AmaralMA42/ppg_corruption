# CG-RV01 — Pendencias de engenharia SEM patches aplicados nesta branch

**Escopo:** testes e propostas, nunca reescrita espontanea de simulacao; Work local pode estar em versao posterior ao GitHub master de 25/06/2026.

| ID | P | Situacao do pacote CG-RV01 | Acao futura | Aceite minimo |
|---|---|---|---|---|
| P01 | 0 | Ja implementado no pacote: payoff de estrategia ausente => NaN, medias condicionais | NAO reaplicar; conferir consumidores antigos | present-only correto; social ponderado inalterado |
| P02 | 1 | Protocolo ACF_QUARTER_V2 criado apos diagnosticos E3 | Preservar posterioridade, versao e janelas; distinguir censura | teste de lag >10000 e nota de mudanca de protocolo |
| P03 | 0 | 18 origens de paineis incompletas | Recuperar dados ou sinalizar figuras como substitutas | tabela file/hash/config/seed/plot/script por figura |
| P04 | 2 | tex com label eq:oscillatory_point problematica | Somente proposta editorial humana | referencia compilada e denominacao calibrada |
| P05 | 1 | comparacao antiga L100/L400 com alpha=0.35 vs ancora alpha=0.40 | Nao atribuir parametros do ponto A a outros arquivos | legenda traz parametros do arquivo de origem |
| P06 | 0 | absorbing_window pode confundir inatividade temporaria com absorcao | Teste de monocultura exata separado de contador de inatividade | nenhuma configuracao mista e declarada absorvente exata por silencio finito |
| P07 | 0 | motor assume 4 vizinhos e 5 grupos mas expõe G ajustavel | Validar G=5 ou implementar generalizacao real | G!=5 rejeitado explicitamente ou provado correto |
| P08 | 0 | leitor de NPZ antigo nao trata 0 marcador de ausencia | Adaptador legacy por rho(t)>0, recalculo de medias | 1 replica all-C + 1 replica all-D no formato antigo reproduz payoff condicionado 11.5 e social 5.75 no teste referencia |
| P09 | 1 | replotagem usa cfg corrente, nao cfg salvo | Reconstituir config a partir de metadados do NPZ | mudanca de cfg nao muda r/sigma/alpha declarados na figura arquivada |
| P10 | 1 | SEM com n=1 pode parecer precisamente zero | usar NA e rotulo n=1 | sem barra de incerteza n=1 rotulada SEM=0 |
| P11 | 2 | Github legado diagnostic_payoff.py afirma falsamente que D paga c | Corrigir apenas diagnostico legado, nao kernel | D paga custo 0 no PGG C/D; teste correto |

## Ordem recomendada quando houver autorizacao de modificar codigo
1. Registrar SHA do checkout local e comparar com `CG_RV01_PACOTE_GPT.zip`; usar a versao local de producao como fonte, nao sobrescrever a partir desta branch de documentacao.
2. Testar P06, P07 e P08 isoladamente; corrigir apenas falhas reproduzidas.
3. Testar replotagem/meta (P09) e SEM n=1 (P10).
4. Rodar `python src/safety_gate.py` sempre que alterar fisica, RNG, paralelizacao, payoff ou vizinhanca (conforme AGENTS.md do repo).
5. Conservar hashes, outputs negativos, backcompat e resultados de regressao. Nenhuma execucao estatistica adicional foi feita por esta consolidacao documental.

**Status desta branch:** NO_CODE_WRITES. Nao usar estes registros como commit de reparos do motor.
