# Registo de decisões

Por ordem cronológica. Cada entrada: **o quê** e **porquê**.

---

### 2026-10-29 23:58 · Ambiente
- **O quê:** venv local (`.venv`, Python 3.10.12) com PyTorch 2.6.0+cu124, scikit-learn 1.7.2, umap-learn 0.5.12, scikit-image 0.25.2, matplotlib 3.10, pandas 2.3. Versões completas em `outputs/env.txt`.
- **Porquê:** o driver (CUDA 13.0) suporta as wheels cu124; esta versão do PyTorch já está validada nesta máquina noutros projetos, o que reduz o risco de incompatibilidades.

### 2026-10-29 23:59 · Máquina partilhada: limites de memória
- **O quê:** GPU única (RTX 2080 Ti, 11 GB). No arranque havia outro treino (de outro projeto) a ocupar cerca de 4.4 GB e 94% da GPU, e a RAM tinha só cerca de 0.5 a 1.8 GB disponíveis (15.5 GB no total, quase tudo ocupado por outras aplicações; swap quase cheia).
- **Porquê/consequências:** (1) a sondagem de memória limita a fração de memória GPU do próprio processo (`set_per_process_memory_fraction`, deixando 1.5 GB de margem) para nunca provocar OOM nos processos alheios; (2) o número de treinos em paralelo é limitado pela **RAM** (cerca de 1 GB de RSS por processo com contexto CUDA), não pela memória GPU; (3) KernelPCA corre em float32 para gastar metade da RAM; (4) tempos por época são ruidosos porque a GPU e os CPUs são partilhados.

### 2026-10-30 00:05 · Leitura dos dados
- **O quê:** `dados/` tem os 4 ficheiros IDX (`*-ubyte.gz`). Leitura direta com `gzip` + `struct` (validando o magic number), sem dependências externas. `dados/` nunca é escrito.
- **Porquê:** é o formato original do Fashion-MNIST; ler diretamente evita downloads e garante que os dados são exatamente os fornecidos.

### 2026-10-30 00:06 · Splits e normalização
- **O quê:** `train_test_split` estratificado (seed 42): 60k de treino → 54 000 train / 6 000 val (600 por classe); os 10k de teste ficam intocados até à avaliação final. Índices guardados em `outputs/splits/` (train/val relativos aos 60k originais). Pixels divididos por 255 → [0,1], float32, guardados em duas versões: flatten `[N,784]` e imagem `[N,1,28,28]` (`outputs/data/`).
- **Porquê:** estratificar mantém as classes perfeitamente balanceadas em todos os splits; guardar índices torna os splits reprodutíveis e auditáveis; as duas versões evitam reshape repetidos (PCA/kNN usam flatten, CNNs usam imagem).

### 2026-10-30 00:06 · Subconjunto de visualização
- **O quê:** 10 000 índices estratificados **do split train** (`outputs/splits/viz10k_idx.npy`), usados em t-SNE/UMAP (Fase 3) e nas grelhas/métricas de agrupamento dos embeddings (Fase 4).
- **Porquê:** as visualizações são exploratórias e os métodos (t-SNE, UMAP, AEs) não usam labels, por isso não há fuga de informação ao usar o train; o test continua reservado para as tabelas finais. Usar sempre os mesmos 10k pontos torna todas as figuras comparáveis entre si.

### 2026-10-30 00:07 · kNN em GPU
- **O quê:** kNN implementado com `torch.cdist` em blocos de 1024 queries contra a galeria inteira em GPU; voto maioritário uniforme, empates resolvidos pela classe de menor índice (o mesmo critério do scikit-learn). Distâncias euclidiana e cosseno.
- **Porquê:** dezenas de espaços × 54k pontos tornam o kNN de CPU lento; em GPU cada avaliação demora segundos. Validado: kNN no PCA com os 784 componentes dá exatamente a mesma accuracy que em pixels (0.8613), como deve ser (rotação + translação preservam distâncias).

### 2026-10-30 00:10 · EDA: critério para a hipótese das classes de tronco
- **O quê:** a hipótese "Shirt/T-shirt/Pullover/Coat são as mais confundidas" considera-se confirmada se os 4 pares de classes mais confundidos pelo kNN (k=5) em pixels forem todos entre essas quatro classes. Registo também a fração de todos os erros que são trocas entre elas.
- **Porquê:** um primeiro critério (as 4 classes com pior accuracy serem exatamente estas) era demasiado rígido: Sandal aparece em 4.º lugar por ser confundida com Sneaker, o que não contradiz a hipótese. Avaliar por pares de confusão é o que a hipótese realmente afirma.

### 2026-10-30 00:12 · SSIM vetorizado em GPU
- **O quê:** SSIM implementado em PyTorch reproduzindo os valores por omissão de `skimage.metrics.structural_similarity` (janela uniforme 7×7, covariância amostral, K1=0.01, K2=0.03, `data_range=1`, média só na região válida).
- **Porquê:** o SSIM é preciso em milhares de pares (métricas dos AEs) e em query × galeria inteira (proxy de similaridade da Fase 6, cerca de 27M pares). Verificado contra o scikit-image em 200 pares aleatórios: diferença máxima 4.7e-5 (só precisão float32).

### 2026-10-30 00:13 · PCA
- **O quê:** `PCA(svd_solver="covariance_eigh")`, ajustado só no train. Um modelo por (k, whitening) guardado em joblib e projeções train/val/test em `.npy` float32. Para a curva de saturação usa-se o PCA completo truncado.
- **Porquê:** com 54 000 × 784, decompor a matriz de covariância 784×784 é exato e demora cerca de 4 s, em vez de uma SVD completa dos dados.
- **Saturação:** definida como o menor k cuja accuracy kNN (val) fica a ≤ 0.5 p.p. do máximo da curva.

### 2026-10-30 00:13 · PCA por classe
- **O quê:** um PCA completo por classe (ajustado nos 5 400 exemplos de treino da classe); o MSE de reconstrução por classe usa o PCA da classe verdadeira.
- **Porquê:** mede quanto se ganha por modelar cada classe separadamente. Nota: é um cenário "oráculo" (exige conhecer a classe), útil como comparação, não como método de redução prático.

### 2026-10-30 00:13 · IncrementalPCA, KernelPCA, LDA, TwoNN
- **IncrementalPCA:** 128 componentes, batches de 2 048 (sempre ≥ n_components). Compara-se com o PCA exato pela variância acumulada, MSE, ângulo máximo entre subespaços e kNN.
- **KernelPCA RBF:** subconjunto estratificado de 10 000 exemplos do train (índices guardados), 128 componentes, `eigen_solver="randomized"`, γ pela heurística da mediana (γ = 1/mediana‖xᵢ−xⱼ‖²). A projeção do train completo, val e test é feita em blocos de 4 000 para limitar a RAM. Não se ajusta `fit_inverse_transform` (exigiria outro sistema 10k×10k em RAM e a reconstrução não é pedida para o KernelPCA).
- **LDA:** `solver="svd"` (robusto aos pixels de borda com variância nula), 9 componentes (máximo para 10 classes). Comparada com PCA de 1 a 9 dimensões em kNN e regressão logística.
- **TwoNN:** estimador de Facco et al. (2017) com descarte dos 10% maiores rácios μ = r₂/r₁ (como no artigo) e remoção de duplicados exatos (r₁ = 0); reporto também a versão de máxima verosimilhança.

### 2026-10-30 00:15 · Arquiteturas dos autoencoders
- **DenseAE:** 784-512-256-d / d-256-512-784, ReLU, saída sigmoide (cerca de 1.1M parâmetros).
- **ConvAE:** conv 3×3 stride 2 (28→14→7), canais 32-64-64, BatchNorm+ReLU, Linear 3136→d; decoder espelhado com ConvTranspose (0.13M a 0.92M parâmetros, consoante d).
- **TransformerAE:** patches 4×4 → 49 tokens, embedding linear para 128 dims + embeddings posicionais aprendidos, 4 blocos pre-LN (4 cabeças, MLP 256, GELU), *attention pooling* com uma query aprendida → Linear para d. Decoder simétrico: z → Linear para 128, replicado pelos 49 tokens + embeddings posicionais próprios do decoder, 4 blocos, Linear por token para o patch 4×4, sigmoide (cerca de 1.15M parâmetros).
- **Porquê:** capacidades parecidas (cerca de 1M parâmetros, exceto o ConvAE com d pequeno, que é naturalmente mais compacto); **nenhuma skip connection**: o decoder só recebe z (mais constantes aprendidas), por isso toda a informação passa pelo bottleneck. Attention pooling em vez de média simples porque a média dos tokens dilui a informação posicional necessária à reconstrução.

### 2026-10-30 00:15 · Protocolo de treino comum
- **O quê:** MSE, AdamW (lr 1e-3, weight decay 0), warmup linear de 500 passos, redução do LR para metade após 3 épocas sem melhoria (ReduceLROnPlateau implementado à mão para coexistir com o warmup), early stopping com patience 8 no MSE de validação, máximo de 60 épocas, seed 42 antes de construir cada modelo; guarda-se o checkpoint com o melhor MSE de validação.
- **Porquê:** a única coisa que muda entre modelos é a arquitetura (e d). O warmup é necessário para o Transformer arrancar de forma estável e é inócuo para os outros, por isso aplica-se a todos e não é preciso um LR específico por arquitetura. 60 épocas com patience 8 chega para as curvas estabilizarem (ver históricos) mantendo a grelha em cerca de 1 h.
- **Retoma:** `last.pt` gravado no fim de cada época (modelo, otimizador, estado do gerador do loader, histórico); se o processo morrer, o treino continua da última época. `last.pt` é apagado quando o modelo termina; `best.pt` fica.
- **Loader:** os três splits ficam inteiros na GPU (cerca de 220 MB) e os mini-batches saem por indexação de uma permutação gerada na própria GPU (`GPUBatchLoader`), em vez do `torch.utils.data.DataLoader`, que faria indexação e colagem item a item em CPU.

### 2026-10-30 00:20 · Batch size, AMP e paralelismo (com base em `gpu_probe.csv` e `batch_convergence.csv`)
- **Batch size = 256 para as três arquiteturas.**
  - A memória não é o fator limitante: a 256 os picos são 0.25 GB (Dense), 0.36 GB (Conv) e 0.94 GB (Transformer), menos de 12% do limite de 8 GB. O "maior batch que deixa 15–20% livre" seria cerca de 2048 no Transformer, 8192 no Conv e > 32k no Dense.
  - Mas batches grandes prejudicam claramente a convergência com o mesmo orçamento de épocas: após 5 épocas (d=32) o val MSE do Transformer é 0.016 a bs 256 e 0.106 a bs 2048; no ConvAE 0.0090 contra 0.041. Uma parte desta diferença vem do warmup fixo de 500 passos (a bs 2048 isso equivale a cerca de 19 épocas), mas mesmo escalando o LR não se compensam 8× menos atualizações em 60 épocas.
  - Acima de bs 512 o tempo por época quase não desce (Transformer 9.5 → 7.5 → 7.2 s; Conv 1.46 → 1.39 → 1.14 s): a GPU já está saturada, por isso não há ganho de tempo que justifique o risco.
  - bs 128 converge um pouco melhor por época, mas demora cerca de 2× mais por época; com early stopping e 60 épocas, 256 é o melhor compromisso. Usar o mesmo batch em todas as arquiteturas mantém a comparação justa, e o LR (1e-3) não precisa de ajuste.
- **Sem AMP (fp32).** A RTX 2080 Ti (Turing, sm_75) não tem bf16 nativo (o PyTorch reporta suporte, mas por emulação). O fp16 foi medido: a bs 256 é mais lento nas três arquiteturas (passo 3.7 → 6.4 ms Dense, 6.9 → 8.8 ms Conv, 45 → 67 ms Transformer), porque os modelos são pequenos e o custo dominante é o lançamento de kernels; só compensaria no Transformer a bs ≥ 1024, que já foi excluído.
- **2 treinos em paralelo na mesma GPU.** Cada processo ocupa cerca de 1 GB de RAM (contexto CUDA + dados) e a máquina só tinha cerca de 1–2 GB livres; com 2 processos a memória GPU total fica abaixo de 2 GB. A fila intercala arquiteturas (Transformer, Conv, Dense para cada d) para que modelos leves corram ao lado dos pesados. A CNN supervisionada entra na mesma fila.

### 2026-10-30 00:24 · CNN supervisionada (teto de referência)
- **O quê:** 5 convoluções 3×3 (32-32-64-64-128) com BatchNorm, 2 max-pool, dropout 0.1–0.3, global average pooling, Linear 128→10; AdamW (lr 1e-3, wd 1e-4), cosine annealing, bs 256, 40 épocas no máximo, early stopping (patience 6) na accuracy de validação; sem data augmentation.
- **Porquê:** é uma CNN simples e padrão; o objetivo é ter um teto de referência para "quanto se consegue com supervisão ponta a ponta", não maximizar a accuracy. Sem augmentation, para treinar com exatamente os mesmos dados que as representações não supervisionadas.

### 2026-10-30 00:25 · Picos de loss no TransformerAE
- **O quê:** com LR 1e-3 o TransformerAE d=2 teve picos pontuais de loss (épocas 9–11 e 15). Mantive o protocolo comum, sem LR específico nem gradient clipping.
- **Porquê:** o ReduceLROnPlateau corta o LR logo a seguir e o modelo recupera e melhora (val MSE 0.0269 na época 12, contra 0.0282 antes do pico); o early stopping guarda sempre o melhor checkpoint. Mudar o protocolo a meio obrigaria a re-treinar tudo e quebraria a regra "só a arquitetura muda".

### 2026-10-30 00:22 · Fase 3 em CPU
- **O quê:** t-SNE (scikit-learn, Barnes-Hut, `init="pca"`, `learning_rate="auto"`, 1000 iterações) e UMAP (`min_dist=0.1`, `random_state=42`) correm com `CUDA_VISIBLE_DEVICES=""` e 4 threads, lendo o train por mmap.
- **Porquê:** com 2 treinos na GPU sobravam cerca de 400 MB de RAM; sem contexto CUDA o processo gasta cerca de 600 MB a menos. As métricas (trustworthiness/continuity/preservação kNN) usam a mesma implementação em PyTorch, a correr em CPU.
- **Métricas:** trustworthiness e continuity (Venna & Kaski) com k=10, implementadas por blocos (evita a matriz 10k×10k de postos do scikit-learn, cerca de 1.6 GB). Validado contra `sklearn.manifold.trustworthiness` em 2 000 pontos: diferença cerca de 1e-7. Continuity = trustworthiness com os papéis trocados. Também registo a preservação de vizinhança (|N₁₀ᵒʳⁱᵍ ∩ N₁₀ᵉᵐᵇ|/10) e a accuracy kNN 2D leave-one-out.

### 2026-10-30 00:30 · Definições das métricas de agrupamento, classificação e retrieval
- **Agrupamento (Fase 4):** silhouette, Davies-Bouldin e NMI/ARI do k-means (k=10, 10 inicializações) calculados no latente original (não no UMAP), nos 10k pontos de visualização; accuracy kNN (k=5) com o train completo como referência e o val como queries. Todas estas métricas são invariantes a uma mudança de escala global, por isso comparam modelos com latentes de escalas diferentes. Ranking final = posto médio nas 5 métricas. Também registo a silhouette no UMAP 2D, como complemento visual.
- **Classificação (Fase 5):** kNN (k=5), regressão logística (StandardScaler + LBFGS, C=1, 1000 iterações) e um MLP pequeno (1 camada escondida de 256, dropout 0.2, Adam 1e-3, early stopping na accuracy de validação), todos com os mesmos hiperparâmetros em todos os espaços. Nota: a regressão logística com StandardScaler é invariante ao whitening do PCA (o whitening é só uma escala por componente), portanto as diferenças com/sem whitening só aparecem no kNN e no MLP. Saturação: menor d com accuracy (val) a ≤ 0.5 p.p. do máximo da família.
- **Retrieval (Fase 6):** galeria = train (54k), queries = 2 000 imagens estratificadas do test; distâncias euclidiana e cosseno em GPU. precision@k = fração do top-k com a classe da query; **class-consistency@k** = fração de queries em que a classe maioritária do top-k é a da query; mAP = média da average precision sobre o ranking completo da galeria (5 400 relevantes por query); mAP@k = AP truncada em k e normalizada por k. As labels só entram nestas métricas, nunca no ranking.
- **Proxy de similaridade visual sem labels:** em 500 das queries (estratificadas), SSIM e IoU de silhueta (pixel > 0.1) contra a galeria inteira. Por espaço: SSIM/IoU médios do top-k, NDCG@10 com ganho = SSIM (ou IoU), em que o ranking ideal é calculado sobre a galeria inteira, e correlação de Spearman entre −distância e SSIM/IoU num conjunto fixo de 1 000 imagens aleatórias por query. Depois mede-se a correlação (Spearman/Kendall) entre o ranking dos espaços por accuracy de classificação e o ranking pela proxy visual.
- **Descritores clássicos:** HOG (9 orientações, células 4×4, blocos 2×2, L2-Hys → 1 296 dims), magnitude Sobel (784), LBP uniforme P=8, R=1 em histogramas por célula numa grelha 4×4 (160) e silhueta binarizada (784). O LDA e o embedding da CNN usam labels no ajuste da representação; entram no retrieval só como referência e marcados como "supervised".
- **Falhas (Fase 8):** em raw, PCA-64, ConvAE d=32 e HOG: (a) pares test→train cujo vizinho mais próximo tem outra classe, ordenados pela distância (sem repetir a imagem do train); (b) por classe, o par do test mais afastado. A causa provável é atribuída por regras simples e explícitas (ver docstring de `src/failures.py`), sem intervenção manual.

### 2026-10-30 01:25 · Orçamento de 60 épocas mantido (sem re-treino mais longo)
- **O quê:** nenhum AE parou por early stopping (todos chegaram às 60 épocas com a melhor época entre 58 e 60). Mantive os modelos tal como estão e não prolonguei o treino.
- **Porquê:** a regra pedida é igual para todos (mesmo nº máximo de épocas, mesma patience) e foi cumprida; os ganhos que ainda restam são pequenos (a melhoria por época no fim é < 1% do MSE) e afetam as três arquiteturas na mesma direção. Prolongar só alguns modelos quebraria a comparação, e prolongar todos custaria cerca de mais 1.5 h de GPU (dominado pelo Transformer). A conclusão é apresentada como "a orçamento fixo de 60 épocas" no relatório.

### 2026-10-30 01:45 · Descritores clássicos também na classificação
- **O quê:** HOG, Sobel, LBP e silhueta foram acrescentados à Fase 5 como família "descritor" (extra, com os mesmos três classificadores).
- **Porquê:** a comparação "melhor para classificar vs melhor para retrieval" só é justa se todos os espaços do retrieval tiverem accuracy de classificação; sem isto, as perspetivas *texture/edges* ficavam de fora.

### 2026-10-30 01:50 · Correlação entre rankings de espaços: subconjuntos e circularidade
- **O quê:** as correlações (Spearman/Kendall) entre accuracy/mAP e a proxy visual são reportadas para todos os espaços e sem os espaços com d=2; a silhueta binarizada é excluída das correlações com a proxy IoU.
- **Porquê:** os espaços d=2 são maus em tudo e inflacionam artificialmente qualquer correlação positiva; a IoU de silhueta usa exatamente a mesma binarização do espaço "silhouette", o que seria circular. Também fica registado que o SSIM (calculado em pixels) favorece um pouco o espaço raw.

### 2026-10-30 02:00 · Relatório
- **O quê:** o notebook só lê CSV/PNG/NPY de `outputs/`; as curvas de treino são desenhadas a partir dos `history.csv`. É executado com `nbconvert --execute`, para que as tabelas e figuras fiquem guardadas no próprio ficheiro.
- **Porquê:** cumpre a regra "o relatório nunca re-treina" e permite ler o relatório sem voltar a correr nada.

---

## Adenda: autoencoders especializados (Conv-VAE e Conv-DAE)

### 2026-10-30 10:20 · Hipóteses (escritas antes de treinar os modelos e de ver qualquer resultado deles)
Critérios de decisão fixados à partida; o veredicto (confirmada / refutada / inconclusiva) fica no fim desta adenda e no relatório.
- **H1:** o VAE tem MSE de reconstrução pior que o ConvAE ao mesmo d, mas latente mais suave (melhor em interpolação, possivelmente melhor ou igual em retrieval).
  - (a) MSE (val) do `vae_conv` > MSE do `conv_ae` em pelo menos 5 dos 6 valores de d.
  - (b) Interpolação (d=16 como caso principal, d=64 como confirmação), em 200 pares aleatórios do val: o VAE tem (i) menor coeficiente de variação do tamanho dos passos entre frames consecutivos (transição mais uniforme, sem saltos) e (ii) menor rácio "distância ao vizinho real mais próximo dos frames intermédios / mesma distância nos extremos" (os frames intermédios são tão plausíveis como as próprias reconstruções).
  - (c) Retrieval, parte "possivelmente": mAP e precision@10 do VAE ≥ ConvAE − 1 p.p. ao mesmo d. Não conta para o veredicto principal.
  - Confirmada se (a) e (b) se verificarem; refutada se (a) ou (b) falhar claramente; inconclusiva se os resultados de (b) divergirem entre d=16 e d=64 ou entre as duas medidas.
- **H2:** o DAE iguala ou melhora o ConvAE em accuracy de classificação e é claramente mais robusto a queries ruidosas.
  - (a) accuracy test (kNN e LogReg) do `dae_conv` ≥ `conv_ae` − 0.5 p.p. em pelo menos 5 dos 6 d (por classificador).
  - (b) Na experiência de robustez (d=64), a precision@10 do DAE com σ = 0.3 e com σ = 0.5 é pelo menos 3 p.p. superior à do ConvAE base, e a queda face a σ = 0 é menor.
  - Confirmada se (a) e (b) se verificarem; parcialmente confirmada se só uma.
- **H3:** o melhor espaço para classificação não coincide com o melhor para a proxy de similaridade visual.
  - O espaço com maior accuracy LogReg (val) não é o de maior NDCG@10-SSIM, e a correlação de Spearman entre os rankings dos espaços (accuracy vs Spearman(−dist, SSIM), sem d=2) tem |ρ| < 0.3, agora com os 12 espaços especializados incluídos.
  - **Nota de honestidade:** esta hipótese já tinha suporte nos espaços base (Fase 6: ρ = 0.03). A adenda testa se se mantém com os especializados, não é um teste "às cegas".

### 2026-10-30 10:20 · Arquitetura dos modelos especializados
- **`dae_conv`:** exatamente a arquitetura do ConvAE base (mesma classe, mesmas camadas e parâmetros); só muda o input de treino.
- **`vae_conv`:** o mesmo tronco convolucional e o mesmo decoder do ConvAE; a única diferença é a cabeça do encoder: a `Linear(3136→d)` do ConvAE passa a ser a cabeça de **mu** e acrescenta-se uma segunda `Linear(3136→d)` para **logvar**. Com a mesma seed, as camadas comuns começam com os mesmos pesos iniciais do ConvAE. Custo: +3136·d + d parâmetros (de +9.4k com d=2 a +401k com d=128).
- `encode()` do VAE devolve **mu** (determinístico), por isso classificação, retrieval, UMAP, métricas e `ae_encode` usam mu sem código especial; logvar fica guardado à parte (`outputs/latents/vae_conv_d{d}_logvar_{split}.npy`). As reconstruções de avaliação do VAE são descodificadas a partir de mu.
- **Nomes:** `vae_conv_d{d}` e `dae_conv_d{d}`, como pedido; ficam numa lista própria em `config.yaml` (`ae.specialized`), separada da grelha base, para os scripts das fases poderem distinguir "base" de "especializado".

### 2026-10-30 10:20 · Protocolo de treino dos especializados
- **Igual à grelha base:** AdamW (lr 1e-3, wd 0), warmup de 500 passos, ReduceLROnPlateau (×0.5, patience 3), early stopping (patience 8), máximo de 60 épocas, batch 256, seed 42, mesmo split, fp32.
- **VAE:** loss = Σ_pixels (x − x̂)² por imagem (média no batch) + β·KL(q(z|x) ‖ N(0, I)), **β = 1**; KL warm-up linear por passo durante as **primeiras 10 épocas** (β_t = β·min(1, passo / (10 × passos por época))).
  - O LR não muda apesar de a loss estar somada por imagem (escala cerca de 784× maior): o Adam é invariante a uma escala constante da loss.
  - Nota: MSE somado com β = 1 equivale a um decoder gaussiano com σ² = 0.5, o que é uma pressão forte do KL face ao erro em pixels. Espera-se que várias dimensões fiquem inativas (é exatamente o que a contagem de unidades ativas mede); mantenho β = 1, que é o valor pedido.
  - Critério de early stopping e de escolha do melhor checkpoint: loss de validação com β = 1 (MSE somado com reconstrução a partir de mu + KL), ou seja, o objetivo final. Durante as 10 épocas de warm-up, os contadores de patience e de plateau não avançam, porque o objetivo ainda está a mudar.
  - **Unidades ativas:** dimensões j com Var_x[mu_j] > 0.01 (critério de Burda et al., 2016), calculadas no val e no test.
- **DAE:** input = clip(x + 0.3·ε, 0, 1) com ε amostrado de N(0, I), novo em cada batch (gerador próprio com seed 42, guardado no checkpoint para a retoma), alvo = x limpo, loss MSE (média por pixel, como no base). **σ = 0.3.**
  - Early stopping e melhor checkpoint: MSE de denoising no val com um ruído fixo (seed 42), o mesmo objetivo do treino. O MSE com input limpo também fica no histórico.
  - Avaliação, latentes e métricas de reconstrução: input **limpo** (exceto na experiência de robustez).

### 2026-10-30 10:20 · Análises novas
- **Amostras do VAE:** 100 vetores z amostrados de N(0, I) (seed 42) descodificados numa grelha 10×10 para d ∈ {2, 16, 64}; para d=2 também a "variedade" descodificada numa grelha de quantis da normal (extra).
- **Interpolação:** 10 frames com t de 0 a 1, linear no latente (mu no VAE), para pixels (cross-fade), PCA (k = d), ConvAE, VAE e DAE. Exemplo visual Sneaker → Ankle boot, mais as medidas de H1(b) em 200 pares aleatórios do val. A interpolação em PCA é um cross-fade das duas reconstruções (o decoder do PCA é afim), por isso os frames intermédios são sobreposições "fantasma"; as medidas mostram até que ponto os decoders não lineares fogem disso.
- **Robustez:** queries = as 10 000 imagens do test com ruído gaussiano σ_teste ∈ {0, 0.1, 0.3, 0.5} (clip a [0,1], seed 42, o mesmo ruído para todos os espaços); galeria = train limpo. Métricas: precision@10 (euclidiana) e accuracy kNN (k=5), em raw, PCA-64, `conv_ae_d64`, `dae_conv_d64` e `vae_conv_d64`. Extra: MSE da reconstrução contra a imagem limpa (capacidade de denoising de cada modelo).
- **Fase 3:** UMAP (n_neighbors ∈ {5, 15, 50}) dos latentes de `vae_conv` e `dae_conv` com d=16 e d=64, e também do `conv_ae` com os mesmos d, como referência ao mesmo d (na Fase 3 base o AE era o d=32). Sem t-SNE nestes (pedido: UMAP).
- **Fase 8:** `vae_conv_d32` e `dae_conv_d32` entram na mineração, comparados com `conv_ae_d32`: taxa de erro 1-NN, pares de classes trocados (contagens de todos os erros 1-NN) e sobreposição (Jaccard) dos conjuntos de imagens falhadas.

### 2026-10-30 19:15 · Correção ao código do veredicto de H1 (depois de ver resultados)
- **O quê:** a primeira versão de `hypotheses()` exigia as duas medidas de interpolação em conjunto por d, e por isso devolvia "refutada" quando elas divergiam. Corrigi-a para seguir o texto registado às 10:20 ("inconclusiva se os resultados de (b) divergirem [...] entre as duas medidas").
- **Porquê:** o critério válido é o escrito antes dos resultados; a correção repõe esse critério, não o altera. Fica registada porque foi feita depois de ver os números.

### 2026-10-30 19:15 · Correção a um texto anterior da Fase 6
- **O quê:** o relatório dizia que o cosseno "só piora nos espaços com d=2". Isso estava errado já com os espaços base: também piora na silhueta (−2.8 p.p.) e, de forma marginal, no HOG, na LBP e no TransformerAE d=8. Texto corrigido.
- **Também atualizado:** com os 12 espaços novos, os postos médios e as correlações entre espaços mudam, por serem relativos ao conjunto de espaços. O TransformerAE d=16 passa a 1.º e a LDA-9 a 2.º (a LDA fica em 30.º no kNN); ρ(LogReg, Spearman-SSIM) passa de 0.03 a 0.11. Os valores só com os espaços base estão nas entradas anteriores do `LOG.md`.

### 2026-10-30 19:15 · Veredicto das hipóteses
Números em `outputs/metrics/hypotheses.csv`.
- **H1: inconclusiva.** (a) cumprido: MSE do VAE pior em 6/6 d (d=128: 0.0145 contra 0.0028). (b) as medidas divergem nos dois d: rácio interior/extremos melhor (d=16: 1.100 contra 1.165; d=64: 1.083 contra 1.104), CV dos passos pior (0.382 contra 0.106; 0.422 contra 0.125). (c, secundário) não cumprido em nenhum d (mAP de 0.395 contra 0.520 em d=8; P@10 de 0.794 contra 0.839 em d=64). Limitação: o rácio não separa bem os casos (cross-fade 1.115, melhor do que o ConvAE d=16) e a distância L2 favorece imagens desfocadas.
- **H2: parcialmente confirmada.** (a) refutado: dentro de 0.5 p.p. do ConvAE em 0/6 d no kNN e 1/6 na LogReg. (b) confirmado: P@10 de 0.799 contra 0.530 com σ = 0.3 (+26.9 p.p.) e 0.721 contra 0.193 com σ = 0.5 (+52.8 p.p.); quedas de 1.7 contra 30.6 p.p. e 9.5 contra 64.4 p.p. Os DAEs usaram as 60 épocas com a melhor na última, portanto (a) é a orçamento fixo.
- **H3: confirmada**, sem ser um teste às cegas. `cnn` (LogReg val 0.942) ≠ `conv_ae_d128` (NDCG-SSIM 0.947); ρ = 0.107 (p = 0.49, 44 espaços). A conclusão depende das medidas: com o kNN ρ = 0.38 (p = 0.01); contra o NDCG-SSIM do top-10, ρ = 0.59 (LogReg) e 0.68 (kNN).

### 2026-10-30 19:15 · β mantido em 1, apesar do colapso parcial
- **O quê:** o warm-up do KL não evitou o colapso parcial (16/32, 14/64 e 12/128 unidades ativas) e o MSE fica parado em cerca de 0.0145 a partir de d=8. Não re-treinei com outro β nem com outra escala de reconstrução.
- **Porquê:** β = 1 com MSE somado é o que foi pedido, e mudar depois de ver os resultados deixaria de testar a hipótese fixada. É uma limitação a ter em conta: as conclusões sobre o VAE valem para esta configuração, não para VAEs em geral.

### 2026-10-30 20:43 · Origem de cada resultado no relatório
- **O quê:** novo `src/provenance.py`, que gera `provenance.csv`, `scripts.csv`, `requisitos.csv` e `docs/PROVENIENCIA.md`. O notebook lê estes CSV para pôr em cada figura e tabela o script, a função, a linha, o cálculo, os dados e o comando. Cada fase ganhou um bloco "Método e código" e o relatório ganhou os Anexos A a D. Passaram a aparecer o teste de GPU, os painéis de reconstrução com d=8, 32 e 128, a CNN em detalhe, a accuracy por classe e o retrieval com todos os k.
- **Porquê:** quem lê o relatório tem de conseguir chegar ao código de qualquer resultado. O script e a função são indicados à mão, porque um nome de ficheiro nem sempre identifica o cálculo, mas a linha é procurada no código de cada vez, para não ficar desatualizada. O script falha se faltar a origem de um ficheiro ou se uma função citada não existir, e o notebook falha se uma figura ou tabela ficar por mostrar.
- **Limite:** a verificação confirma que as funções e os ficheiros existem e que a escrita do artefacto está na função indicada. O texto que descreve cada cálculo foi escrito à mão a partir do código e não é verificado automaticamente.

### 2026-10-30 21:37 · PCA por classe fora do relatório; títulos e índice
- **O quê:** a secção do PCA por classe saiu do relatório (tabela, figura, método e interpretação). `src/pca_lda.py::step_per_class`, `pca_per_class.csv`, `pca_per_class.png` e `pca_per_class.joblib` mantêm-se. Os títulos passaram a ser numerados por fase, a subsecção 4.x passou a 4.4 e o relatório abre com um índice das fases.
- **Porquê:** foi pedido retirar a secção do relatório. O enunciado pede "PCA por classe vs global" na Fase 2, por isso o cálculo continua no projeto e o Anexo A assinala que não é mostrado.
