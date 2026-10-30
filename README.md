# Projeto Prático 2: Redução de Dimensionalidade (Fashion-MNIST)

PCA (com e sem whitening, por classe, incremental, kernel), LDA, dimensão intrínseca (TwoNN),
t-SNE/UMAP, autoencoders (Dense, Conv, Transformer) com latentes d ∈ {2, 8, 16, 32, 64, 128},
dois autoencoders especializados (Conv-VAE e Conv-DAE, mesma grelha de d), classificação e retrieval
em todos os espaços, robustez a queries ruidosas e análise de falhas.

> A extensão de cor (Fase 7) **não** foi usada: só se trabalha com o Fashion-MNIST em `dados/`.

## Como correr

```bash
python3 -m venv .venv && .venv/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install scikit-learn umap-learn matplotlib scikit-image pandas pyyaml joblib scipy nbformat nbconvert ipykernel
./run_all.sh                 # tudo, do zero até ao notebook executado (JOBS=2 treinos em paralelo)
```

Cada passo é idempotente (salta o que já está em `outputs/`) e retomável: os treinos gravam
`last.pt` em cada época e continuam de onde pararam. Para refazer um passo, apaga o artefacto
correspondente em `outputs/`. Os passos também podem correr isoladamente:

| Passo | Comando | Saídas principais |
|---|---|---|
| Fase 0: splits e normalização | `python -m src.data` | `outputs/splits/`, `outputs/data/` |
| Fase 1: EDA | `python -m src.eda` | `outputs/metrics/eda_*`, `outputs/figures/eda_*` |
| Smoke test / sondagem de GPU | `python -m src.train_ae --smoke` / `--probe` | `smoke_test.csv`, `gpu_probe.csv`, `batch_convergence.csv` |
| Fase 2: PCA, IPCA, KPCA, LDA, TwoNN | `python -m src.pca_lda` | `outputs/pca/`, `outputs/lda/`, `outputs/metrics/pca_*` |
| Fase 4: grelha de AEs (+ CNN) | `python -m src.train_ae --grid --jobs 2 --with-cnn` | `outputs/ae/`, `outputs/latents/`, `outputs/cnn/` |
| Um só AE | `python -m src.train_ae --arch conv_ae --dim 32` | `outputs/ae/conv_ae_d32/` |
| Adenda: grelha dos especializados | `python -m src.train_ae --grid --specialized` | `outputs/ae/{vae,dae}_conv_d*/` |
| Adenda: amostras, interpolação, painéis | `python -m src.specialized` | `specialized_summary.csv`, `interpolation.csv` |
| Adenda: robustez a ruído | `python -m src.robustness` | `robustness.csv`, `robustness.png` |
| Adenda: veredicto H1–H3 | `python -m src.specialized --hypotheses` | `hypotheses.csv` |
| Fase 3: t-SNE / UMAP | `python -m src.viz_nonlinear` | `outputs/viz/`, `viz_quality.csv` |
| Fase 4: reconstruções e embeddings | `python -m src.embeddings_viz` | `recon_summary.csv`, `embedding_clustering.csv` |
| Fase 5: classificação | `python -m src.classification` | `classification.csv`, `saturation.csv` |
| Fase 6: retrieval | `python -m src.retrieval` | `retrieval.csv`, `retrieval_vs_classification.csv` |
| Fase 8: falhas | `python -m src.failures` | `outputs/failures/` |
| Proveniência | `python -m src.provenance` | `provenance.csv`, `scripts.csv`, `requisitos.csv`, `docs/PROVENIENCIA.md` |
| Fase 9: relatório | `jupyter nbconvert --execute --inplace notebooks/report.ipynb` | `notebooks/report.ipynb` |

### Autoencoders especializados (adenda)

Usam exatamente o encoder/decoder do ConvAE base e o mesmo protocolo de treino; só muda o objetivo.

- **`vae_conv` (Conv-VAE):** o encoder produz mu e logvar; treino com MSE somado por imagem + β·KL (β = 1,
  warm-up linear do KL nas 10 primeiras épocas). Serve para ter um latente regular (próximo de N(0, I)),
  que permite gerar amostras e interpolar de forma suave. Como latente (classificação, retrieval, UMAP)
  usa-se **mu**; logvar fica em `outputs/latents/vae_conv_d<d>_logvar_*.npy`.
- **`dae_conv` (Conv-DAE):** treino com input ruidoso (gaussiano, σ = 0.3, clip a [0,1]) e alvo limpo.
  Serve para obter representações robustas a ruído; avaliado com input limpo, exceto na experiência
  de robustez.

```bash
python src/train_ae.py --arch vae_conv --dim 32     # equivalente: python -m src.train_ae --arch vae_conv --dim 32
python src/train_ae.py --arch dae_conv --dim 32
```

### Retrieval a partir de código

```python
from src.retrieval import retrieve, available_spaces
from src.data import load_split
X, y = load_split("test")
res = retrieve(X[0], space="conv_ae_d32", k=10, metric="cosine")   # índices no split train + distâncias
available_spaces()   # raw, silhouette, pca_k64, pca_k64_whiten, sobel, hog, lbp, dense_ae_d16, ...,
                     # vae_conv_d32, dae_conv_d32, ..., lda, cnn
```

### Funções de representação

```python
from src.representation import pca_project, pca_reconstruct, ae_encode, ae_decode
Z = pca_project(X, k=64);  Xr = pca_reconstruct(Z, k=64)
z = ae_encode(X, "transformer_ae", 32);  Xr = ae_decode(z, "transformer_ae", 32)
```

## Onde é calculado cada resultado

Cada figura e tabela do relatório tem uma legenda com o script, a função, a linha, o cálculo, os dados de entrada e o
comando que a refaz. Os índices completos estão em `docs/PROVENIENCIA.md` e nos Anexos A a D do relatório; o Anexo A liga
cada requisito do enunciado e da adenda ao código. O `src/provenance.py` procura as linhas no código de cada vez e falha se
um ficheiro de `outputs/` ficar sem origem ou se uma função citada não existir.

## Estrutura

```
dados/                 dados originais (IDX .gz), só leitura
config.yaml            configuração central (seed, splits, hiperparâmetros de todas as fases)
run_all.sh             pipeline completo
src/
  utils.py             config, seeds, logging, manifesto de artefactos
  data.py              leitura IDX, normalização [0,1], splits estratificados
  evaluate.py          kNN em GPU, métricas (classificação, trustworthiness/continuity, agrupamento, TwoNN, SSIM/IoU)
  representation.py    pca_project / pca_reconstruct / ae_encode / ae_decode
  models.py            DenseAE, ConvAE, TransformerAE, ConvVAE, CNN supervisionada
  train_ae.py          treino parametrizado (--arch --dim), grelha, smoke test, sondagem de GPU (inclui VAE/DAE)
  train_cnn.py         CNN supervisionada (teto de referência)
  eda.py               Fase 1
  pca_lda.py           Fase 2
  viz_nonlinear.py     Fase 3
  embeddings_viz.py    Fase 4 (reconstruções, grelha UMAP dos embeddings, agrupamento)
  classification.py    Fase 5
  retrieval.py         Fase 6 (retrieve(query, space, k, metric) + avaliação)
  failures.py          Fase 8
  specialized.py       adenda: VAE/DAE (amostras, interpolação, painéis, veredicto das hipóteses)
  robustness.py        adenda: queries ruidosas vs galeria limpa
  provenance.py        origem de cada artefacto e mapa de scripts e requisitos
  plots.py             estilo das figuras
notebooks/report.ipynb relatório final (só lê outputs/)
outputs/
  splits/ data/        índices dos splits; imagens normalizadas (.npy)
  pca/ lda/            modelos (joblib) e projeções train/val/test (.npy float32)
  ae/<arch>_d<d>/      best.pt, history.csv, metrics.json
  latents/             <arch>_d<d>_{train,val,test}.npy (float32); no VAE = mu, mais <..>_logvar_*.npy
  cnn/                 CNN supervisionada: best.pt, history.csv, metrics.json, embeddings
  features/            descritores HOG, Sobel, LBP, silhueta
  viz/                 coordenadas 2D (t-SNE, UMAP, embeddings)
  retrieval/ failures/ índices de queries, matrizes da proxy visual, casos de falha
  metrics/             todas as tabelas (CSV)
  figures/             todas as figuras (PNG)
  logs/                um log por script/treino
  manifest.json        índice de todos os artefactos (caminho, descrição, data, config)
  env.txt              versões dos pacotes e GPU
docs/DECISIONS.md      decisões e justificações (cronológico)
docs/LOG.md            diário de execução e resultados-chave
docs/PROVENIENCIA.md   scripts, requisitos e origem de cada figura e tabela
```

## Reprodutibilidade

- Seed global **42** (Python, NumPy, PyTorch) definida antes de cada split, ajuste e construção de modelo.
- Splits estratificados guardados como índices; o test (10k) só é usado nas tabelas finais.
- Toda a configuração está em `config.yaml`; versões em `outputs/env.txt`.
- Hardware usado: 1 × RTX 2080 Ti (11 GB), partilhada com outros processos (ver `docs/LOG.md`).
  Mesmo com seed fixa, os kernels CUDA não são todos determinísticos, por isso podem surgir
  diferenças pequenas (na 3.ª ou 4.ª casa decimal) entre execuções.
