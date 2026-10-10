# CG-RV01 — Experimento proposto e ADIADO (nao executado)

**Questao causal:** a transferencia *seletiva* para C explica a coexistencia observada ou retornos de recursos nao seletivos com a mesma contabilidade geram fenomeno semelhante?

## Pre-contrato (rascunho; depende de decisao humana)
- Preservar o mecanismo baseline em FULL: extracao ANTES da multiplicacao (`sigma` fixo), custo c, regra de Fermi, grupos G=5, rede quadrada periodica, K e seeds.
- Criar um controle de destino **nao seletivo**, por exemplo distribuir a mesma fracão alpha E entre todos os jogadores nao-P elegiveis no grupo (C e D), mantendo a retencao total do corrupto constante. Definir precisamente regras de elegibilidade quando nao ha destinatarios; evitar fonte/sumidouro oculto e contabilizar por grupo.
- Criar teste de invariancia do ledger em todas as composicoes nC+nD+nP=5; garantir que diferenca de welfare nao decorra de mudanca nao intencional do total distribuido/retido.
- Verificar payoffs em all-C, all-D, all-P e misturas de dois e tres tipos, inclusive sem C ou sem nao-P. Reportar quais controles mudam a dominancia P-D sem C.
- Rodar pequeno piloto pareado de tamanhos, seeds e horizonte se autorizado: comparadores FULL versus controle contabilidade-fechada; registrar coexistencia, extincoes, densidades, payoff social, invasoes nas interfaces e tempo de absorcao **exata**, nao simples silencio.
- Calcular IC, incluir desfechos negativos; condicionar em sobreviventes apenas em analise secundária.
- Rodar `safety_gate.py` e testes de contabilidade antes de qualquer simulacao cientifica.

## Falsificador central
Se o controle nao seletivo reproduzir coexistencia, interfaces e payoffs dentro das incertezas, a seletividade a C **nao sera necessaria para o fenomeno naquele regime**. A seletividade pode ainda afetar outros parametros, velocidades ou transientes. Se houver diferenca, examinar quais gradientes de payoff e interfaces a produzem antes de reivindicar um mecanismo exclusivo.

**Decisao 10/10/2026:** primeiro consolidar registros duros e compartilhar com outros chats; nao rodar este experimento nem editar .tex nesta etapa.
