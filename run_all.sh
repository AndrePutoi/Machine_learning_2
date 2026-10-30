#!/usr/bin/env bash
# Corre o projeto inteiro a partir de dados/. Cada passo é idempotente: o que já existe em outputs/
# é saltado, por isso pode ser interrompido e relançado em qualquer altura.
#   ./run_all.sh            # tudo
#   JOBS=3 ./run_all.sh     # mais treinos em paralelo na grelha de AEs
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-2}
mkdir -p outputs/logs

(echo "# Ambiente ($(date '+%Y-%m-%d %H:%M'))"; $PY --version; \
 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true; \
 $PY -m pip freeze) > outputs/env.txt

$PY -m src.data                                   # Fase 0: splits e normalização
$PY -m src.eda                                    # Fase 1: EDA mínima
$PY -m src.train_ae --smoke                       # validação rápida do código dos AEs
[ -f outputs/metrics/gpu_probe.csv ] || $PY -m src.train_ae --probe   # sondagem de batch/memória
$PY -m src.pca_lda                                # Fase 2: PCA, whitening, IPCA, KPCA, LDA, TwoNN
$PY -m src.train_ae --grid --jobs "$JOBS" --with-cnn   # Fase 4: 18 AEs + CNN supervisionada
$PY -m src.train_ae --smoke --specialized         # adenda: validação rápida do VAE/DAE
$PY -m src.train_ae --grid --jobs "$JOBS" --specialized   # adenda: 12 AEs especializados (VAE/DAE)
$PY -m src.train_ae --collect
$PY -m src.viz_nonlinear                          # Fase 3: t-SNE/UMAP (raw, PCA-50, ConvAE) + UMAP VAE/DAE
$PY -m src.embeddings_viz                         # Fase 4: reconstruções, grelha de embeddings, agrupamento
$PY -m src.classification                         # Fase 5: classificação em todos os espaços
$PY -m src.retrieval                              # Fase 6: retrieval e proxy visual
$PY -m src.failures                               # Fase 8: análise de falhas
$PY -m src.robustness                             # adenda: robustez a queries ruidosas
$PY -m src.specialized                            # adenda: amostras/interpolação do VAE, painéis, tabela
$PY -m src.specialized --hypotheses               # adenda: veredicto das hipóteses H1-H3
$PY -m src.provenance                            # Fase 9: origem de cada figura e tabela, verificada no código
$PY -m jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb   # Fase 9
