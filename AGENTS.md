Para qualquer mudanca que toque a simulacao fisica, paralelizacao, payoff,
vizinhança, seeds, sweeps ou otimizacao de runtime, rode antes de finalizar:

```bash
python src/safety_gate.py
```

Se o ambiente Windows bloquear `python`, use a venv local:

```bash
.venv\Scripts\python.exe src\safety_gate.py
```

Nao trate uma otimizacao como pronta enquanto esse gate nao passar. Alteracoes
em documentacao, artigo, figuras estaticas ou comentarios que nao afetem codigo
executavel nao precisam desse gate.
