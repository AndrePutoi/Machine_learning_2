# Diário de execução

## 2026-10-29 23:58 · Setup e GPU
- `nvidia-smi`: 1 × **RTX 2080 Ti** (11 GB, driver 580.126, CUDA 13.0). No arranque: 5.4 GB ocupados e 94% de utilização por outro treino (não deste projeto); terminou por volta das 00:13, ficando cerca de 9.5 GB livres.
- RAM: 15.5 GB no total, apenas cerca de 0.5 a 1.8 GB disponíveis durante a sessão (restante ocupado por outras aplicações). 12 CPUs, também partilhados.
- Instalado: PyTorch 2.6.0+cu124, scikit-learn 1.7.2, umap-learn 0.5.12, scikit-image 0.25.2 (`outputs/env.txt`).

## 2026-10-30 00:08 · Fase 0
- Dados IDX lidos de `dados/`: train 60 000 × 28 × 28, test 10 000; 6 000 por classe.
- Splits estratificados (seed 42): train 54 000 (5 400/classe), val 6 000 (600/classe), test 10 000 (1 000/classe). Índices em `outputs/splits/`; imagens normalizadas em `outputs/data/`.

## 2026-10-30 00:09 · Smoke test dos AEs (1 época, d=2 e d=128)
- Todas as arquiteturas treinam e as formas de encode/decode estão corretas (`outputs/metrics/smoke_test.csv`).
- Tempo/época (bs 256, GPU ainda partilhada): DenseAE 0.7 s, ConvAE 2.2 s, TransformerAE 12–14 s.

## 2026-10-30 00:11 · Fase 1 (EDA)
- kNN (k=5) em pixels raw, train → val: **accuracy 0.861**, macro-F1 0.861.
- Accuracy por classe: Shirt 0.64, Coat 0.74, Pullover 0.82, Sandal 0.84, Dress 0.84, T-shirt 0.89; Trouser/Sneaker/Bag/Ankle boot ≥ 0.96.
- Pares mais confundidos: Pullover↔Coat (146), T-shirt↔Shirt (144), Pullover↔Shirt (118), Coat↔Shirt (83), Sandal↔Sneaker (50).
- **Hipótese confirmada:** os 4 pares mais confundidos são todos entre classes de tronco; 60.9% de todos os erros são trocas entre T-shirt/Pullover/Coat/Shirt.
- Distâncias em pixels: a distância intra-classe (6.9 a 10.9) é sempre menor do que a inter-classe média (11.0 a 12.4), mas Shirt é a classe com intra-classe mais alta entre as de tronco (9.5) e a menor inter-classe média (11.0): está "no meio" das outras. Classe mais próxima: Pullover↔Coat, Shirt→Coat, Sandal↔Sneaker, Ankle boot→Sneaker.

## 2026-10-30 00:13–00:20 · Sondagem de GPU (`outputs/metrics/gpu_probe.csv`, `batch_convergence.csv`)
- Limite imposto ao processo: 8.0 GB (memória livre − 1.5 GB de margem).
- Pico de memória (d=128) a bs 256: DenseAE 0.25 GB, ConvAE 0.36 GB, TransformerAE 0.94 GB. OOM só no TransformerAE a bs 4096 (bs 2048 = 5.8 GB); ConvAE chega a bs 16384 (8.0 GB, no limite); DenseAE cabe até 32768 (0.9 GB).
- Tempo/época: acima de bs 512 quase não desce (TransformerAE 9.5 s a 256 → 7.5 s a 512 → 7.2 s a 2048; ConvAE 1.46 → 1.39 → 1.14 s).
- Convergência (val MSE após 5 épocas, d=32): TransformerAE 0.013 (128), 0.016 (256), 0.029 (512), 0.069 (1024), 0.106 (2048); ConvAE 0.0079, 0.0090, 0.0117, 0.0176, 0.041; DenseAE 0.0126, 0.0154, 0.0194, 0.028, 0.052.
- AMP fp16 a bs 256 é **mais lento** nas três arquiteturas (passo de 3.7 → 6.4 ms no DenseAE, 6.9 → 8.8 ms no ConvAE, 45 → 67 ms no TransformerAE); só compensa no TransformerAE a bs 1024.
- Decisão: bs 256, fp32, 2 treinos em paralelo (ver DECISIONS).

## 2026-10-30 00:13–00:19 · Fase 2 (PCA e afins)
- PCA (train): componentes para **80% = 24, 90% = 84, 95% = 188, 99% = 459**. PC1 explica 29.0%, PC2 17.7%.
- MSE de reconstrução (val): k=2 0.0459, 8 0.0264, 16 0.0203, 32 0.0150, 64 0.0102, 128 0.0062.
- kNN (k=5, val) sem / com whitening: k=2 0.525/0.526, 8 0.796/0.796, 16 0.847/0.850, 32 0.867/0.868, **64 0.873**/0.867, 128 0.872/0.858. O whitening só ajuda (pouco) até k=32; a partir daí amplifica componentes de baixa variância (ruído) e a accuracy desce.
- Curva fina kNN vs k: satura em **k = 40** (≤ 0.5 p.p. do máximo 0.873, em k=64); com 784 componentes volta a 0.861 (= pixels raw, como esperado).
- PCA por classe: n_95 varia de 74 (Trouser) a 221 (Sandal) e 205 (Bag); global = 188. Com o PCA da própria classe (oráculo) o MSE (val) com k=32 desce de 0.0150 para 0.0095.
- IncrementalPCA (128 comp., batches 2048): variância acumulada e MSE praticamente iguais aos do PCA exato (diferença < 0.1% na variância, MSE 0.00623 vs 0.00620 para k=128); kNN idêntico até k=64. Ângulo máximo entre subespaços ≈ 0° até k=16 e 1.4° em k=64; em k=128 sobe para 72° porque os últimos componentes têm valores próprios quase iguais (a fronteira do subespaço fica mal definida), não por erro do método. Custo: 98 s contra 1.7 s do PCA exato.
- KernelPCA RBF (10k pontos, γ = 0.00756 pela mediana, ajuste 17 s): kNN val 0.512 (k=2), 0.790 (8), 0.833 (16), 0.863 (32), 0.870 (64), **0.873** (128). Igual ao PCA linear em accuracy máxima, mas precisa de mais componentes para lá chegar.
- LDA (9 comp.): bate o PCA com o mesmo nº de dimensões em quase todos os casos. Com 9 dims: kNN 0.836 vs 0.811 (PCA-9), LogReg 0.841 vs 0.763. Com 2 dims: kNN 0.563 vs 0.525.
- TwoNN (train): dimensão intrínseca **global ≈ 15.7** (MLE 14.2); por classe de 12.7 (Trouser) e 13.2 (Sandal) a 16.4 (Shirt e Sneaker). Bem abaixo do k de saturação do kNN-PCA (40): o PCA linear precisa de mais dimensões do que a dimensão intrínseca da variedade não linear onde os dados vivem.

## 2026-10-30 00:20 · Grelha de treino lançada
- 18 AEs + CNN, 2 em paralelo, em background (`python -m src.train_ae --grid --jobs 2 --with-cnn`; logs em `outputs/logs/ae_*.log`, `cnn.log`, `ae_grid.log`).
- Estimativa antes do lançamento: TransformerAE com cerca de 10–12 s/época × ≤60 épocas × 6 ≈ 60–70 min; ConvAE ≈ 12 min; DenseAE ≈ 5 min; CNN ≈ 3 min. Com 2 trabalhos em paralelo, **cerca de 45–60 min** no total.
- Correção da estimativa (00:30): com 2 trabalhos em paralelo o TransformerAE corre a cerca de 17 s/época (e não 10–12 s), mas dois Transformers em simultâneo mantêm esse ritmo cada um. Nova previsão de fim da grelha: por volta das 01:35.

## 2026-10-30 00:24 · CNN supervisionada (teto)
- 40 épocas (melhor na 39), 3.0 min, 141k parâmetros, pico 0.82 GB. **val acc 0.9455, test acc 0.933** (macro-F1 0.933).

## 2026-10-30 00:22–00:31 · Fase 3 (t-SNE/UMAP em raw e PCA-50; o ConvAE d=32 entra quando estiver treinado)
- t-SNE: trustworthiness 0.989–0.990 (raw) e 0.994–0.995 (PCA-50); preservação kNN@10 0.39–0.42 (raw) e 0.45–0.48 (PCA-50); melhor perplexity = 30.
- UMAP: trustworthiness 0.978–0.982 (raw) e 0.984–0.988 (PCA-50), mas continuity um pouco maior do que no t-SNE (até 0.990 com n_neighbors=50); preservação kNN 0.22–0.35, a descer com n_neighbors.
- Leitura: o t-SNE preserva melhor as vizinhanças locais; o UMAP com n_neighbors grande dá mais peso à estrutura global (clusters mais compactos e separados) à custa da vizinhança fina. Partir do PCA-50 melhora todas as métricas face aos pixels (remove ruído de baixa variância). Em todos os mapas, Trouser, Bag e o grupo do calçado separam-se bem, enquanto T-shirt/Pullover/Coat/Shirt (e parte de Dress) formam uma região contínua sobreposta, coerente com a EDA.
- Accuracy kNN no próprio mapa 2D (leave-one-out): 0.81–0.82 com t-SNE e 0.75–0.78 com UMAP.

## 2026-10-30 00:40–00:42 · Descritores para retrieval
- Silhueta (784) 5 s, Sobel (784) 5 s, HOG (1 296) 80 s, LBP (160) 15 s, para os 70k exemplos (`outputs/features/`).

## 2026-10-30 01:25 · Grelha de AEs concluída (64.3 min de relógio, 2 em paralelo)
- 18/18 AEs + CNN treinados, sem erros (`outputs/metrics/ae_summary.csv`). **Nenhum modelo parou por early stopping**: todos correram as 60 épocas com a melhor época entre 58 e 60, ou seja, o orçamento de épocas (igual para todos) é o fator limitante e todos ainda melhorariam ligeiramente com mais treino.
- Custo por arquitetura (bs 256, fp32, GPU partilhada entre 2 treinos):

| Arquitetura | Parâmetros | s/época | Tempo por modelo | Tempo total (6 modelos) | Pico memória GPU |
|---|---|---|---|---|---|
| DenseAE | 1.07M a 1.13M | 0.70 | 0.7 min | 4.2 min | 0.26 GB |
| ConvAE | 0.13M a 0.92M | 2.9 | 2.9 min | 17.5 min | 0.52 GB |
| TransformerAE | 1.14M a 1.18M | 15.2 | 15.2 min | 91.1 min | 0.94 GB |
| CNN supervisionada | 0.14M | 4.5 | 3.0 min | – | 0.82 GB |

- MSE de reconstrução (val), d = 2 / 8 / 16 / 32 / 64 / 128:
  - PCA: 0.0459 / 0.0264 / 0.0203 / 0.0150 / 0.0102 / 0.0062
  - DenseAE: 0.0237 / 0.0118 / 0.0093 / 0.0072 / 0.0064 / 0.0059
  - ConvAE: 0.0256 / 0.0116 / 0.0084 / 0.0061 / 0.0043 / **0.0028**
  - TransformerAE: 0.0241 / **0.0109** / **0.0082** / 0.0067 / 0.0055 / 0.0050
- SSIM (val) com d=128: ConvAE 0.924, TransformerAE 0.868, DenseAE 0.841.
- Leitura rápida: com d pequeno (2–16) os três AEs têm erro cerca de 2× menor do que o PCA e ficam próximos entre si (o Dense é o melhor em d=2, o Transformer em d=8–16); com d grande o ConvAE destaca-se (o viés convolucional aproveita bem a capacidade extra), o DenseAE estagna (cerca de 0.006, igual ao PCA-128) e o Transformer fica no meio.
- Fase 3 completada com o latente do ConvAE d=32 (01:05): t-SNE T=0.992–0.994, accuracy kNN 2D 0.84–0.85 (a melhor das três entradas); UMAP T=0.977–0.984, kNN 2D 0.80–0.82.

## 2026-10-30 01:12 · Fase 8 (análise de falhas)
- 88 casos minados automaticamente (22 por espaço em raw, PCA-64, ConvAE d=32, HOG), em `outputs/failures/`.
- Erro do 1-NN no test: raw 15.3%, PCA-64 15.4%, HOG 14.0%, ConvAE d=32 13.1%.
- Vizinhos com classe diferente: sobretudo T-shirt↔Shirt e Pullover↔Coat (pares quase idênticos, provável ambiguidade de anotação); no HOG, Dress↔Trouser. Mesma classe muito afastada: brilho/contraste em raw/PCA; forma/estilo no ConvAE e no HOG.

## 2026-10-30 01:25–01:37 · Fase 4 (análise): reconstruções, embeddings e agrupamento
- Painéis de reconstrução para todos os d e figura MSE/SSIM vs d (`outputs/figures/recon_*`); `recon_summary.csv` com PCA e os 18 AEs.
- Grelha UMAP de todos os modelos + PCA + LDA (`embeddings_grid.png`) e métricas de agrupamento (`embedding_clustering.csv`).
- Ranking (posto médio de silhouette, Davies-Bouldin, kNN, NMI e ARI): 1.º LDA-9 (supervisionada; silhouette 0.30, NMI 0.73), 2.º **TransformerAE d=16** (NMI 0.65, kNN 0.885), 3.º ConvAE d=16, 4.º TransformerAE d=8, 5.º ConvAE d=8. O PCA fica entre 15.º e 26.º. Com d grande o kNN sobe mas a silhouette e o k-means pioram.

## 2026-10-30 00:53–02:00 · Fase 5 (classificação)
- Teto: CNN supervisionada test 0.933. Pixels raw (test): kNN 0.852, LogReg 0.834, MLP 0.885.
- Melhores não supervisionados (test): TransformerAE d=128 kNN **0.885**; ConvAE d=128 MLP **0.902**; HOG MLP 0.903 e LogReg 0.871 (melhor LogReg não supervisionada).
- d=8 (kNN test): AEs 0.845–0.849 contra PCA 0.784. d=2: AEs 0.68–0.72 contra PCA 0.50.
- Saturação no kNN (≤ 0.5 p.p. do máximo): DenseAE/ConvAE d=32, PCA/TransformerAE d=64. LogReg/MLP ainda sobem até 128 (< 1 p.p. a partir de 64).
- Os descritores HOG/Sobel/LBP/silhueta foram acrescentados à tabela (extra) para poderem entrar na comparação com o retrieval. A LogReg é lenta nestes espaços (silhueta 376 s, HOG 570 s).

## 2026-10-30 01:36–02:00 · Fase 6 (retrieval)
- 39 espaços × 2 distâncias (`retrieval.csv`); proxy visual com SSIM/IoU de 500 queries × 54k imagens da galeria (`outputs/retrieval/`).
- Por classe: CNN P@10 0.92 / mAP 0.85; LDA mAP 0.68; melhores não supervisionados ConvAE d=8/16 e TransformerAE d=16 (mAP ≈ 0.52); HOG 0.50; raw 0.45.
- Proxy visual: NDCG@10 (SSIM) máximo no ConvAE d=128 (0.947), raw (0.944) e TransformerAE d=128 (0.942); **CNN 0.65 e LDA 0.64 (os piores, só acima dos espaços d=2)**.
- Correlação entre espaços (sem d=2): accuracy LogReg vs Spearman(−dist, SSIM) ρ = 0.03 (p = 0.87); mAP vs NDCG-SSIM ρ = −0.15 (p = 0.38); accuracy vs NDCG-SSIM ρ = 0.54. → **melhor para classificar ≠ melhor para semelhança visual**.
- O cosseno melhora o mAP em 31/39 espaços; o whitening destrói a ordenação global (PCA-128: mAP 0.45 → 0.19).

## 2026-10-30 02:02 · Fase 9 (relatório)
- `notebooks/report.ipynb` gerado e executado com `jupyter nbconvert --execute` (54 células, 0 erros; só lê `outputs/`). Cada fase tem tabelas, figuras e uma interpretação curta.

---

# Adenda: autoencoders especializados (Conv-VAE e Conv-DAE)

## 2026-10-30 10:15–10:20 · Preparação
- Pré-requisito verificado: os 18 AEs base e a CNN supervisionada estavam treinados e guardados.
- Hipóteses H1–H3 e critérios de decisão escritos em `docs/DECISIONS.md` antes de qualquer treino.
- `src/models.py`: `ConvVAE` (tronco e decoder do ConvAE, cabeças mu/logvar); `dae_conv` = a própria `ConvAE`. Com a mesma seed, as camadas comuns do VAE começam com os mesmos pesos do ConvAE (verificado).
- `src/train_ae.py`: objetivos VAE (MSE somado + β·KL com warm-up de 10 épocas) e DAE (σ = 0.3); flags `--specialized` para a grelha e o smoke test; `python src/train_ae.py ...` também funciona.
- Smoke test (`smoke_test_specialized.csv`): 1 época com d=2 e d=128, cerca de 1.3–3.9 s/época, cerca de 0.5 GB de pico.
- Máquina: GPU quase livre (cerca de 1 GB ocupado por outro processo), RAM disponível cerca de 1 GB com swap muito ocupada → mantive 2 treinos em paralelo.

## 2026-10-30 10:20 · Grelha dos especializados lançada (12 modelos, 2 em paralelo)
- VAE: o KL estabiliza em cerca de 8.8 nats por imagem tanto com d=8 como com d=16 e o MSE fica parecido (val 0.0143 vs 0.0145): com β = 1 e MSE somado, o VAE só usa uma capacidade de cerca de 9 nats, independentemente de d (ver unidades ativas).
- VAE d=8 e d=16 pararam por early stopping (épocas 54 e 42); são os primeiros modelos do projeto a parar antes das 60 épocas.

## 2026-10-30 10:35 · Grelha dos especializados concluída (15.3 min, 2 em paralelo)
- 12/12 modelos com pesos, histórico, latentes (VAE: mu e logvar) e métricas; sem `last.pt` a sobrar nem erros.
- Por arquitetura: VAE 2.25 s/época, 12.0 min no total, 0.52 GB, 4/6 com early stopping (épocas 54, 42, 57, 47). DAE 2.10 s/época, 12.6 min, 0.54 GB, todos com 60 épocas (melhor época = 60).
- MSE val (d=2/8/16/32/64/128): ConvAE 0.0256/0.0116/0.0084/0.0061/0.0043/0.0028; VAE 0.0264/0.0143/0.0145/0.0145/0.0145/0.0145; DAE 0.0268/0.0132/0.0099/0.0081/0.0068/0.0061.
- VAE: KL 5.1 (d=2) e cerca de 8.7 nats nos restantes; unidades ativas 2, 8, 16, 16, 14, 12.

## 2026-10-30 até 19:12 · Integração nas Fases 3–8 e análises novas
- **Robustez (`robustness.csv`), P@10 com σ = 0/0.1/0.3/0.5:** ConvAE-64 0.837/0.814/0.530/0.193; DAE-64 0.816/0.814/0.799/0.721; VAE-64 0.795/0.777/0.589/0.296; raw 0.803/0.797/0.757/0.620; PCA-64 0.806/0.801/0.759/0.581.
- **Interpolação (`interpolation.csv`):** ver o veredicto de H1 em `DECISIONS.md`.
- **Fase 3:** 18 UMAP novos (ConvAE/VAE/DAE × d=16/64 × 3 n_neighbors); T 0.97–0.99, C 0.98–0.99.
- **Fase 4:** ranking de 38 espaços (coluna `specialized`): 1.º TransformerAE d=16, 2.º LDA-9, 3.º ConvAE d=16; melhor DAE (d=16) em 10.º; melhor VAE (d=8) em 18.º; PCA entre 20.º e 38.º.
- **Fase 5 (test):** kNN máx. ConvAE 0.883, DAE 0.863, VAE 0.847; MLP 0.902, 0.887, 0.885.
- **Fase 6:** 51 espaços × 2 distâncias; cosseno melhora o mAP em 41/51; ρ(LogReg, Spearman-SSIM) sem d=2 = 0.11.
- **Fase 8:** 132 casos minados (6 espaços); erro 1-NN: ConvAE-32 13.1%, DAE-32 15.2%, VAE-32 18.0%; Jaccard com o ConvAE: DAE 0.55, VAE 0.43; calçado 9.9% → 14.7% dos erros no VAE.
- **Hipóteses (`hypotheses.csv`):** H1 inconclusiva, H2 parcialmente confirmada, H3 confirmada.
- Figuras corrigidas depois de as rever: legenda do ranking, etiquetas de `retrieval_vs_classification`, linha cortada do raw em `robustness.png`, linha vazia em `denoise_panel.png`.

## 2026-10-30 20:43 · Relatório com a origem de cada resultado
- `python -m src.provenance`: 620 artefactos, 22 scripts e 51 requisitos verificados (`provenance.csv`, `scripts.csv`, `requisitos.csv`, `docs/PROVENIENCIA.md`).
- `notebooks/report.ipynb`: 56 figuras (53 de `outputs/figures/` e 3 desenhadas no notebook) e todas as tabelas CSV/JSON de `outputs/metrics/` e `outputs/failures/` com legenda de origem. Nada foi treinado nem recalculado.

## 2026-10-30 21:37 · Relatório: índice das fases e títulos
- Índice das fases no início, títulos numerados e PCA por classe fora do relatório (continua calculado em `outputs/`). Nada foi treinado nem recalculado.
