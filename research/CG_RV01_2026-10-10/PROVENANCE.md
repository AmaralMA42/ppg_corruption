# CG-RV01 — Fontes, checksums, limites de acesso e cadeia de custodia

**Data:** 2026-10-10. **Status:** registro documental; integracao de conhecimento sem troca de codigo de producao ou texto do artigo.

## Fontes entregues em conversas anteriores
- `CG_RV01_PACOTE_GPT.zip`, 46.754.007 bytes, SHA256 `412a70a79b711364e82d3f9922346810bd3929053e90dd0a071094f475df81db`. Contem projeto local, relatorios, CSV, JSON, 240 NPZ E5 e copia de `.tex` / `.bib`.
- `CG_RV01_REVISAO_INDEPENDENTE_2026-10-10.md`, SHA256 `ab7bc4ab66ff671c126a3fe3cec7646ae1f30ba944510b1993e756a00ad9c479`.
- `MISSION_CG_RV01_SCIENTIFIC_REPAIR_VALIDATION_v1.0.md`, SHA256 `52d0f3ec751e7d6d9e6d68964d663eaef0c591fc1419b30edab8d855133095da`.
- `DOCUMENTO_ONTOLOGICO.md`, SHA256 `14d74be9d3b1df63581b236fe023b50c4f1a9c8544fdc16e4a62ea7b19db3b2d`.
- `CG_RV01_KNOWLEDGE_FREEZE_NONCANONICAL_2026-10-10.zip`, SHA256 `d39436331e0254dcfb4a0da7451b4e5c78272da896cf92899bd824b9cd21d98c`. Contem 00..06, incluido ledger CSV e manifesto com hashes de codigo.

## Identificadores de arquivos cientificos incluidos no ZIP principal
- `project/corruption_games_PRE_enriched.tex`: SHA256 `cfb0709984fad6e4e9dc8f1717e506df553c15822f6106ecb8a4ab00e400ea36`.
- `project/src/core_simulation.py`: SHA256 `752a52aef2de939e5b3a0f8fffc4449265d831ca12ff6e08a8b05b4748ccc9e1`.
- `project/src/payoff_observables.py`: SHA256 `3d50e58af47618b80a9f88f4e3b45eab3db7f553e53feb84154b3f02c15a3ecf`.
- `project/oscillation_campaign_kit/campaigns/CG_RV01/continuation/reports/CURRENT_STATUS.md`: SHA256 `31ed96feebbc86af54e89ad3014001c3e49d9541a2ed91cba65dc54ec2aab514`.
- `project/oscillation_campaign_kit/campaigns/CG_RV01/continuation/reports/SCIENTIFIC_REVIEW.md`: SHA256 `d8b207336361a908fc41fbe2df2a12335f86aed924141332de52baba61870b55`.
- `project/oscillation_campaign_kit/campaigns/CG_RV01/continuation/reports/CAUSAL_ABLATIONS.md`: SHA256 `0fa361f19bde1c2e564c0235d4d134430ae478f1d43dd3c22145eee67e983c31`.

## Procedimentos efetivamente realizados
Em revisao anterior da mesma conversa: verificacao de 1275 hashes no ZIP principal, reanalise dos 240 arquivos NPZ do E5, conferindo hashes/soma de densidades/ledger e payoff social, e execucao do `src/safety_gate.py` (sete grupos PASS em Linux). Na congelacao documental seguinte: inspectores textuais e programas, sem rerun E5. Nesta integracao GitHub/Notion: transcricao estruturada de resultados ja registrados; **NAO foram feitas novas simulacoes ou releitura integral de referencias externas**.

## Limites de auditabilidade
- Oito ZIPs complementares citados em manifestos nao acompanharam o pacote; nao se pode certificar seus bytes.
- Dezoito linhagens de paineis originais ainda nao tem proveniencia completa.
- Resultados E3 r=3.575 e 3.6 possuem horizontes distintos.
- Os indicadores da campanha de oscilações foram obtidos de um conjunto finito e metodo correlacionado; os 240 registros de sensibilidade sao transformacoes das mesmas seeds, nao 240 replicas.
- A versao GitHub `master` de 25/06/2026 NAO e necessariamente o codigo vigente do Work; esta branch apenas contem documentos criados a partir daquele ancestral, nao atualiza src.
- Este repositorio de resumos **nao substitui os NPZ originais nem o ZIP de 46 MB**. Os valores em CSV aqui sao estatisticas agregadas com definicoes, nao dados brutos por MCS.

## Nao fazer
Nao inferir execucao in situ no Work, nao importar labels como verdade fisica geral, nao usar datas de analise como datas de simulacao, nao chamar bloco documental de patch aplicado e nao promover conteudo deste pacote ao manuscrito automaticamente.
