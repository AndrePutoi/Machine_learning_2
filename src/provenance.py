"""Proveniência: onde é calculado cada artefacto e onde está cada requisito do enunciado no código.

Gera, sem treinar nem recalcular nada:
  outputs/metrics/provenance.csv   artefacto -> script, função, linha, comando, cálculo, dados de entrada
  outputs/metrics/scripts.csv      um registo por script (papel, funções, dependências, comando)
  outputs/metrics/requisitos.csv   requisito do enunciado/adenda -> código -> artefactos -> secção do relatório
  docs/PROVENIENCIA.md             as mesmas três tabelas para ler fora do notebook

A linha é procurada no código de cada vez, por isso não fica desatualizada. O script falha se:
  - um ficheiro de outputs/ não tiver origem registada;
  - uma regra não corresponder a nenhum ficheiro;
  - uma função citada (`src/x.py::f`) não existir ou a escrita do artefacto não estiver nessa função;
  - um script de src/ não estiver no mapa.

Uso: python -m src.provenance
"""
from __future__ import annotations

import ast
import re
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd

from .utils import OUT, ROOT, get_logger, load_json, out, register

log = get_logger("provenance")
SRC = ROOT / "src"

# Comando que recria os artefactos de cada script
CMD = {
    "src/data.py": "python -m src.data",
    "src/eda.py": "python -m src.eda",
    "src/pca_lda.py": "python -m src.pca_lda",
    "src/train_ae.py": "python -m src.train_ae --grid --jobs 2 --with-cnn",
    "src/train_cnn.py": "python -m src.train_cnn",
    "src/viz_nonlinear.py": "python -m src.viz_nonlinear",
    "src/embeddings_viz.py": "python -m src.embeddings_viz",
    "src/classification.py": "python -m src.classification",
    "src/retrieval.py": "python -m src.retrieval",
    "src/failures.py": "python -m src.failures",
    "src/specialized.py": "python -m src.specialized",
    "src/robustness.py": "python -m src.robustness",
}
ONE_AE = "python -m src.train_ae --arch <arch> --dim <d>"
EN, AD, EX = "enunciado", "adenda", "extra"
TRAIN = "outputs/data/X_train_flat.npy"
VAL = "outputs/data/X_val_flat.npy"
TEST = "outputs/data/X_test_flat.npy"


def A(padrao, script, funcao, fase, origem, calculo, dados, w=None, cmd=None):
    """Uma regra de proveniência. `w` = regex da linha de escrita, dentro de `funcao`."""
    return dict(padrao=padrao, script=f"src/{script}.py", funcao=funcao, fase=fase, origem=origem,
                calculo=calculo, dados=dados, w=w, cmd=cmd)


ARTEFACTOS = [
    # ------------------------------------------------------------------ Fase 0
    A("outputs/splits/viz10k_idx.npy", "data", "prepare", "Fase 0", EN,
      "subconjunto estratificado de 10k índices do split train, para as visualizações", "dados/*-labels-idx1-ubyte.gz",
      r"save_npy\(np\.sort\(viz_rel"),
    A("outputs/splits/*_idx.npy", "data", "prepare", "Fase 0", EN,
      "split estratificado 54k/6k com `train_test_split` (seed 42), em `src/data.py::prepare`; test = os 10k originais",
      "dados/*-labels-idx1-ubyte.gz", r"save_npy\(ix"),
    A("outputs/data/*", "data", "prepare", "Fase 0", EN,
      "leitura IDX (`src/data.py::_read_idx`) e divisão por 255; versões flatten 784 e 1×28×28", "dados/*.gz", r"save_npy\(Xf"),
    # ------------------------------------------------------------------ Fase 1
    A("outputs/figures/eda_mean_images.png", "eda", "main", "Fase 1", EN,
      "média pixel a pixel das imagens de cada classe", TRAIN),
    A("outputs/metrics/eda_class_means.npy", "eda", "main", "Fase 1", EN,
      "média pixel a pixel das imagens de cada classe", TRAIN, r"save_npy\(means"),
    A("outputs/figures/eda_class_distances.png", "eda", "main", "Fase 1", EN,
      "média das distâncias euclidianas entre todos os pares de cada par de classes (`src/eda.py::class_distance_matrix`)", TRAIN),
    A("outputs/metrics/eda_class_distances.csv", "eda", "main", "Fase 1", EN,
      "média das distâncias euclidianas entre todos os pares de cada par de classes (`src/eda.py::class_distance_matrix`)",
      TRAIN, r"to_csv\(files\[0\]"),
    A("outputs/figures/eda_knn_raw_confusion.png", "eda", "main", "Fase 1", EN,
      "kNN k=5 em pixels (`src/evaluate.py::knn_predict`) e matriz de confusão normalizada (`src/evaluate.py::confusion`)", f"{TRAIN}, {VAL}"),
    A("outputs/metrics/eda_knn_raw_confusion.csv", "eda", "main", "Fase 1", EN,
      "contagens da confusão do kNN k=5 em pixels (`src/evaluate.py::confusion`)", f"{TRAIN}, {VAL}", r"to_csv\(files\[1\]"),
    A("outputs/metrics/eda_confused_pairs.csv", "eda", "main", "Fase 1", EN,
      "10 pares de classes com mais confusões, somando as duas direções", "eda_knn_raw_confusion.csv", r"to_csv\(files\[2\]"),
    A("outputs/metrics/eda_summary.json", "eda", "main", "Fase 1", EN,
      "resumo da EDA e critério da hipótese: os 4 pares mais confundidos são todos de tronco", "eda_class_distances.csv, eda_knn_raw_confusion.csv",
      r"save_json\(summary"),
    # ------------------------------------------------------------------ Teste de GPU
    A("outputs/metrics/smoke_test.csv", "train_ae", "smoke", "Teste de GPU", EN,
      "1 época por arquitetura, com d=2 e d=128 (`src/train_ae.py::fit`)", f"{TRAIN}, {VAL}", r"to_csv\(p", "python -m src.train_ae --smoke"),
    A("outputs/metrics/smoke_test_specialized.csv", "train_ae", "smoke", "Teste de GPU", AD,
      "1 época de VAE e DAE, com d=2 e d=128 (`src/train_ae.py::fit`)", f"{TRAIN}, {VAL}", r"to_csv\(p",
      "python -m src.train_ae --smoke --specialized"),
    A("outputs/metrics/gpu_probe.csv", "train_ae", "probe", "Teste de GPU", EN,
      "batch size a duplicar desde 128 até OOM; 25 passos medidos, com e sem AMP a 256 e 1024; `max_memory_allocated`", TRAIN,
      r"df\.to_csv\(p", "python -m src.train_ae --probe"),
    A("outputs/metrics/batch_convergence.csv", "train_ae", "probe", "Teste de GPU", EN,
      "val MSE após 5 épocas (d=32) para batch 128 a 2048, com o mesmo LR", f"{TRAIN}, {VAL}", r"to_csv\(p2", "python -m src.train_ae --probe"),
    # ------------------------------------------------------------------ Fase 2
    A("outputs/pca/pca_full.joblib", "pca_lda", "step_full", "Fase 2", EN,
      "PCA completo ajustado só no train (`src/pca_lda.py::fit_pca`, solver `covariance_eigh`)", TRAIN, r"joblib\.dump\(pca, model_p"),
    A("outputs/figures/pca_variance.png", "pca_lda", "step_full", "Fase 2", EN,
      "`explained_variance_ratio_` do PCA completo e a sua soma acumulada", "outputs/pca/pca_full.joblib"),
    A("outputs/metrics/pca_explained_variance.csv", "pca_lda", "step_full", "Fase 2", EN,
      "`explained_variance_ratio_` do PCA completo e a sua soma acumulada", "outputs/pca/pca_full.joblib", r"to_csv\(ev_p"),
    A("outputs/metrics/pca_components_thresholds.csv", "pca_lda", "step_full", "Fase 2", EN,
      "menor nº de componentes com variância acumulada ≥ 80/90/95/99%", "outputs/pca/pca_full.joblib", r"th\.to_csv"),
    A("outputs/metrics/pca_reconstruction.csv", "pca_lda", "step_reconstruction", "Fase 2", EN,
      "MSE por pixel entre X e X reconstruído com k componentes (`src/pca_lda.py::recon_mse`)", f"outputs/pca/pca_full.joblib, {VAL}, {TEST}", r"df\.to_csv\(p"),
    A("outputs/figures/pca_reconstructions.png", "pca_lda", "step_reconstruction", "Fase 2", EN,
      "reconstruções por classe para cada k (`src/pca_lda.py::recon_mse`)", f"outputs/pca/pca_full.joblib, {VAL}"),
    A("outputs/figures/pca_eigenfashion.png", "pca_lda", "step_reconstruction", "Fase 2", EN,
      "primeiros vetores de `components_` mostrados como imagens 28×28", "outputs/pca/pca_full.joblib"),
    A("outputs/figures/pca_reconstruction_mse.png", "pca_lda", "step_reconstruction", "Fase 2", EN,
      "MSE de reconstrução por k (`src/pca_lda.py::recon_mse`)", "outputs/metrics/pca_reconstruction.csv"),
    A("outputs/pca/pca_k*.joblib", "pca_lda", "step_whitening", "Fase 2", EN,
      "PCA com k componentes, com e sem whitening (`src/pca_lda.py::fit_pca`)", TRAIN, r"joblib\.dump\(pca, mp"),
    A("outputs/pca/proj/pca_k*.npy", "pca_lda", "step_whitening", "Fase 2", EN,
      "`pca.transform` de train/val/test, float32", "outputs/pca/pca_k*.joblib", r"save_npy\(pca\.transform"),
    A("outputs/metrics/pca_knn_whitening.csv", "pca_lda", "step_whitening", "Fase 2", EN,
      "kNN k=5 no espaço PCA, train → val (`src/evaluate.py::knn_predict`, `src/evaluate.py::cls_metrics`)", "outputs/pca/proj/pca_k*_{train,val}.npy", r"df\.to_csv\(p"),
    A("outputs/figures/pca_knn_whitening.png", "pca_lda", "step_whitening", "Fase 2", EN,
      "accuracy kNN por k, com e sem whitening", "outputs/metrics/pca_knn_whitening.csv"),
    A("outputs/metrics/pca_knn_curve.csv", "pca_lda", "step_knn_curve", "Fase 2", EN,
      "kNN k=5 em 21 valores de k; saturado = até 0.5 p.p. do máximo", f"outputs/pca/pca_full.joblib, {TRAIN}, {VAL}", r"df\.to_csv\(p"),
    A("outputs/pca/pca_per_class.joblib", "pca_lda", "step_per_class", "Fase 2", EN,
      "um PCA por classe, ajustado só nas imagens dessa classe do train", TRAIN, r"joblib\.dump\(models"),
    A("outputs/metrics/pca_per_class.csv", "pca_lda", "step_per_class", "Fase 2", EN,
      "componentes para cada limiar e MSE no val com o PCA global e com o da própria classe (oráculo)", f"{TRAIN}, {VAL}", r"df\.to_csv\(p"),
    A("outputs/figures/pca_per_class.png", "pca_lda", "step_per_class", "Fase 2", EN,
      "componentes para 95% e MSE, PCA por classe vs global", "outputs/metrics/pca_per_class.csv"),
    A("outputs/pca/ipca_k128.joblib", "pca_lda", "step_ipca", "Fase 2", EN,
      "`IncrementalPCA` com 128 componentes, `partial_fit` em blocos de 2048", TRAIN, r"joblib\.dump\(ipca"),
    A("outputs/pca/proj/ipca_k128_*.npy", "pca_lda", "step_ipca", "Fase 2", EN,
      "`ipca.transform` de train/val/test", "outputs/pca/ipca_k128.joblib", r"save_npy\(ipca\.transform"),
    A("outputs/metrics/ipca_vs_pca.csv", "pca_lda", "step_ipca", "Fase 2", EN,
      "variância, MSE, ângulo entre subespaços (`subspace_angles`) e kNN, IPCA vs PCA", f"outputs/pca/ipca_k128.joblib, outputs/pca/pca_full.joblib, {VAL}", r"df\.to_csv\(p"),
    A("outputs/pca/kpca_rbf_k128.joblib", "pca_lda", "step_kpca", "Fase 2", EN,
      "`KernelPCA` RBF, 128 componentes, 10k pontos, γ = 1/mediana(‖xᵢ−xⱼ‖²)", TRAIN, r"joblib\.dump\(kpca"),
    A("outputs/pca/kpca_subset_idx.npy", "pca_lda", "step_kpca", "Fase 2", EN,
      "subconjunto estratificado de 10k do train (`src/data.py::stratified_subset`)", "outputs/data/y_train.npy", r"save_npy\(sub"),
    A("outputs/pca/proj/kpca_rbf_k128_*.npy", "pca_lda", "step_kpca", "Fase 2", EN,
      "`kpca.transform` de train/val/test em blocos de 4000", "outputs/pca/kpca_rbf_k128.joblib", r"save_npy\(Z"),
    A("outputs/metrics/kpca_knn.csv", "pca_lda", "step_kpca", "Fase 2", EN,
      "kNN k=5 no espaço KernelPCA, train → val", "outputs/pca/proj/kpca_rbf_k128_{train,val}.npy", r"df\.to_csv\(p"),
    A("outputs/lda/lda.joblib", "pca_lda", "step_lda", "Fase 2", EN,
      "`LinearDiscriminantAnalysis` (9 componentes, solver svd) ajustado no train", f"{TRAIN}, outputs/data/y_train.npy", r"joblib\.dump\(lda"),
    A("outputs/lda/lda_*.npy", "pca_lda", "step_lda", "Fase 2", EN,
      "`lda.transform` de train/val/test", "outputs/lda/lda.joblib", r"save_npy\(lda\.transform"),
    A("outputs/metrics/lda_vs_pca.csv", "pca_lda", "step_lda", "Fase 2", EN,
      "kNN k=5 e LogReg (`src/evaluate.py::logreg`) com 1 a 9 dimensões, LDA vs PCA", "outputs/lda/lda_*.npy, outputs/pca/pca_full.joblib", r"df\.to_csv\(p"),
    A("outputs/metrics/lda_explained_variance.csv", "pca_lda", "step_lda", "Fase 2", EN,
      "`explained_variance_ratio_` do LDA", "outputs/lda/lda.joblib", r"variancia_explicada_lda"),
    A("outputs/figures/lda_vs_pca.png", "pca_lda", "step_lda", "Fase 2", EN,
      "accuracy kNN e LogReg por dimensão, LDA vs PCA", "outputs/metrics/lda_vs_pca.csv"),
    A("outputs/metrics/twonn.csv", "pca_lda", "step_twonn", "Fase 2", EN,
      "TwoNN global e por classe (`src/evaluate.py::twonn`, Facco et al. 2017); k de saturação de `pca_knn_curve.csv`", f"{TRAIN}, outputs/metrics/pca_knn_curve.csv", r"df\.to_csv\(p"),
    A("outputs/figures/twonn.png", "pca_lda", "step_twonn", "Fase 2", EN,
      "dimensão intrínseca TwoNN por classe vs k de saturação", "outputs/metrics/twonn.csv"),
    A("outputs/pca/_done.json", "pca_lda", "main", "Fase 2", EX, "marca de conclusão da fase", "-", r"save_json"),
    # ------------------------------------------------------------------ Fase 3
    A("outputs/viz/tsne_*.npy", "viz_nonlinear", "main", "Fase 3", EN,
      "t-SNE 2D (`src/viz_nonlinear.py::run_tsne`), perplexity 5/30/50, nos 10k pontos", "outputs/splits/viz10k_idx.npy e entrada (`src/viz_nonlinear.py::inputs`)", r"save_npy\(Z, p"),
    A("outputs/viz/umap_*.npy", "viz_nonlinear", "main", "Fase 3", EN,
      "UMAP 2D (`src/viz_nonlinear.py::run_umap`), n_neighbors 5/15/50, min_dist 0.1, nos 10k pontos", "outputs/splits/viz10k_idx.npy e entrada (`src/viz_nonlinear.py::inputs`)", r"save_npy\(Z, p"),
    A("outputs/metrics/viz_quality.csv", "viz_nonlinear", "main", "Fase 3", EN,
      "trustworthiness, continuity e preservação kNN com k=10 (`src/evaluate.py::embedding_quality`); kNN 2D leave-one-out", "outputs/viz/{tsne,umap}_*.npy e a entrada de cada um", r"to_csv\(qual_p"),
    A("outputs/figures/viz_quality_specialized.png", "viz_nonlinear", "specialized_figures", "Fase 3", AD,
      "T, C e preservação kNN do UMAP por n_neighbors, ConvAE vs VAE vs DAE", "outputs/metrics/viz_quality.csv"),
    A("outputs/figures/viz_specialized.png", "viz_nonlinear", "specialized_figures", "Fase 3", AD,
      "UMAP (n_neighbors=15) dos latentes com d=16 e d=64", "outputs/viz/umap_{conv_ae,vae_conv,dae_conv}_d{16,64}_n15.npy"),
    A("outputs/figures/viz_quality.png", "viz_nonlinear", "main", "Fase 3", EN,
      "resumo das métricas de qualidade por método e parâmetro", "outputs/metrics/viz_quality.csv"),
    A("outputs/figures/viz_*.png", "viz_nonlinear", "main", "Fase 3", EN,
      "grelha t-SNE (perplexity) e UMAP (n_neighbors) de uma entrada, com T, C e preservação kNN no título", "outputs/viz/{tsne,umap}_*.npy, outputs/metrics/viz_quality.csv", r'f"viz_\{sp\}'),
    # ------------------------------------------------------------------ Fase 4
    A("outputs/ae/*/best.pt", "train_ae", "fit", "Fase 4", EN,
      "pesos da época com melhor critério de validação (`src/train_ae.py::fit`, `src/train_ae.py::validate`)", f"{TRAIN}, {VAL}", r'_atomic_save\(\{"model"', ONE_AE),
    A("outputs/ae/*/history.csv", "train_ae", "fit", "Fase 4", EN,
      "uma linha por época: loss de treino, val_mse, LR, tempo (VAE: também val_kl e β)", f"{TRAIN}, {VAL}", r'history\.csv', ONE_AE),
    A("outputs/ae/*/metrics.json", "train_ae", "train_one", "Fase 4", EN,
      "MSE/SSIM val e test (`src/train_ae.py::recon_metrics`, `src/evaluate.py::ssim_pairs`), parâmetros, tempo, pico de memória; VAE: `src/train_ae.py::vae_stats`", f"outputs/ae/*/best.pt, {VAL}, {TEST}", r"save_json\(metrics", ONE_AE),
    A("outputs/latents/*_logvar_*.npy", "train_ae", "train_one", "Adenda", AD,
      "logvar do encoder do VAE (`src/train_ae.py::encode_logvar`)", "outputs/ae/vae_conv_d*/best.pt", r"save_npy\(encode_logvar", ONE_AE),
    A("outputs/latents/*.npy", "train_ae", "train_one", "Fase 4", EN,
      "latente de train/val/test (`src/train_ae.py::encode_all`); no VAE é mu", "outputs/ae/*/best.pt", r"save_npy\(encode_all", ONE_AE),
    A("outputs/metrics/ae_summary.csv", "train_ae", "collect", "Fase 4", EN,
      "junta os `metrics.json` dos 30 autoencoders", "outputs/ae/*/metrics.json", r"df\.to_csv\(p", "python -m src.train_ae --collect"),
    A("outputs/cnn/best.pt", "train_cnn", "main", "Fase 5", EN,
      "pesos da época com melhor accuracy de validação", f"{TRAIN}, {VAL}", r'_atomic_save\(\{"model": model\.state_dict\(\), "epoch"'),
    A("outputs/cnn/history.csv", "train_cnn", "main", "Fase 5", EN,
      "uma linha por época: loss, accuracy de validação, LR, tempo", f"{TRAIN}, {VAL}", r'history\.csv'),
    A("outputs/cnn/metrics.json", "train_cnn", "main", "Fase 5", EN,
      "accuracy e macro-F1 em val e test, parâmetros, tempo, memória", f"outputs/cnn/best.pt, {VAL}, {TEST}", r"save_json\(res"),
    A("outputs/cnn/cnn_emb_*.npy", "train_cnn", "main", "Fase 5", EX,
      "embedding de 128 dimensões da penúltima camada (`src/train_cnn.py::predict`)", "outputs/cnn/best.pt", r"save_npy\(emb"),
    A("outputs/metrics/cnn_confusion_*.csv", "train_cnn", "main", "Fase 5", EN,
      "matriz de confusão da CNN (`src/evaluate.py::confusion`)", "outputs/cnn/best.pt", r"pd\.DataFrame\(confusion"),
    A("outputs/metrics/recon_summary.csv", "embeddings_viz", "reconstructions", "Fase 4", EN,
      "MSE/SSIM do PCA (`src/representation.py::pca_recon_from_images`, `src/evaluate.py::ssim_pairs`) ao lado dos AEs", f"{VAL}, {TEST}, outputs/metrics/ae_summary.csv", r"df\.to_csv\(p"),
    A("outputs/figures/recon_mse_ssim.png", "embeddings_viz", "reconstructions", "Fase 4", EN,
      "MSE e SSIM de validação por d", "outputs/metrics/recon_summary.csv"),
    A("outputs/figures/recon_panel_by_d.png", "embeddings_viz", "reconstructions", "Fase 4", EN,
      "reconstruções de 3 classes, para todos os d e modelos (`src/representation.py::ae_reconstruct`)", f"{VAL}, outputs/ae/*/best.pt, outputs/pca/pca_full.joblib"),
    A("outputs/figures/recon_panel_specialized_d*.png", "specialized", "recon_panels", "Adenda", AD,
      "original / PCA / ConvAE / VAE / DAE (`src/representation.py::ae_reconstruct`, `src/representation.py::pca_recon_from_images`)", f"{VAL}, outputs/ae/{{conv_ae,vae_conv,dae_conv}}_d*/best.pt"),
    A("outputs/figures/recon_panel_d*.png", "embeddings_viz", "reconstructions", "Fase 4", EN,
      "original vs PCA vs cada AE, 10 classes (`src/representation.py::ae_reconstruct`, `src/representation.py::pca_recon_from_images`)", f"{VAL}, outputs/ae/*/best.pt, outputs/pca/pca_full.joblib", r"savefig\(fig, name"),
    A("outputs/viz/emb_*.npy", "embeddings_viz", "embed2d", "Fase 4", EN,
      "UMAP 2D do latente (n_neighbors 15, min_dist 0.1); com d=2 usa o latente diretamente", "outputs/latents/*_train.npy, outputs/pca/proj/*, outputs/lda/*", r"save_npy\(E"),
    A("outputs/figures/embeddings_grid.png", "embeddings_viz", "embeddings_and_clustering", "Fase 4", EN,
      "grelha linhas = modelo, colunas = d, sobre os 10k pontos", "outputs/viz/emb_*.npy, outputs/metrics/embedding_clustering.csv"),
    A("outputs/metrics/embedding_clustering.csv", "embeddings_viz", "embeddings_and_clustering", "Fase 4", EN,
      "silhouette, Davies-Bouldin, NMI/ARI k-means (`src/evaluate.py::clustering_metrics`), kNN val e ranking por posto médio", "outputs/latents/*, outputs/pca/proj/*, outputs/lda/*, outputs/splits/viz10k_idx.npy", r"df\.to_csv\(mp"),
    A("outputs/figures/embedding_ranking.png", "embeddings_viz", "embeddings_and_clustering", "Fase 4", EN,
      "espaços ordenados pelo posto médio", "outputs/metrics/embedding_clustering.csv"),
    # ------------------------------------------------------------------ Adenda
    A("outputs/metrics/specialized_summary.csv", "specialized", "summary_table", "Adenda", AD,
      "métricas dos `metrics.json` e MSE de denoising σ=0.3 no test para ConvAE e VAE (`src/representation.py::ae_reconstruct`)", f"outputs/ae/{{conv_ae,vae_conv,dae_conv}}_d*/metrics.json, {TEST}", r"df\.to_csv\(p"),
    A("outputs/figures/denoise_panel.png", "specialized", "recon_panels", "Adenda", AD,
      "reconstrução de imagens com ruído σ=0.3 e 0.5, d=64", f"{VAL}, outputs/ae/{{conv_ae,vae_conv,dae_conv}}_d64/best.pt"),
    A("outputs/figures/vae_samples.png", "specialized", "samples", "Adenda", AD,
      "100 vetores z amostrados de N(0, I) e descodificados (`src/representation.py::ae_decode`); referência: gaussiana ajustada aos latentes do ConvAE", "outputs/ae/vae_conv_d{2,16,64}/best.pt, outputs/latents/conv_ae_d{2,16,64}_train.npy"),
    A("outputs/figures/vae_manifold_d2.png", "specialized", "samples", "Adenda", AD,
      "descodificação de uma grelha 15×15 de quantis 3–97% de N(0, I)", "outputs/ae/vae_conv_d2/best.pt"),
    A("outputs/figures/interpolation_d*.png", "specialized", "interpolation", "Adenda", AD,
      "interpolação Sneaker → Ankle boot em pixels, PCA e latente (`src/specialized.py::_paths`)", f"{VAL}, outputs/ae/*/best.pt, outputs/pca/pca_full.joblib"),
    A("outputs/metrics/interpolation.csv", "specialized", "interpolation", "Adenda", AD,
      "200 pares do val: CV dos passos e distância ao vizinho do train (`src/evaluate.py::knn_search`)", f"{VAL}, {TRAIN}, outputs/ae/*/best.pt", r"df\.to_csv\(p"),
    A("outputs/metrics/robustness.csv", "robustness", "main", "Adenda", AD,
      "queries com ruído e galeria limpa: P@10, kNN e MSE vs imagem limpa (`src/robustness.py::compute`)", f"{TEST}, {TRAIN}, modelos e latentes com d=64 / k=64", r"df\.to_csv\(p"),
    A("outputs/figures/robustness.png", "robustness", "figure", "Adenda", AD,
      "P@10, accuracy kNN e MSE em função de σ", "outputs/metrics/robustness.csv"),
    A("outputs/metrics/hypotheses.csv", "specialized", "hypotheses", "Adenda", AD,
      "critérios H1–H3 fixados em `docs/DECISIONS.md`, avaliados com os CSV das Fases 4–6", "specialized_summary, interpolation, classification, retrieval, robustness, retrieval_vs_classification, retrieval_rank_agreement (CSV)", r"df\.to_csv\(p", "python -m src.specialized --hypotheses"),
    # ------------------------------------------------------------------ Fase 5
    A("outputs/metrics/classification.csv", "classification", "run", "Fase 5", EN,
      "kNN k=5 (`src/evaluate.py::knn_predict`), LogReg (`src/evaluate.py::logreg`) e MLP (`src/classification.py::mlp_fit_predict`) em cada espaço de `src/classification.py::spaces`", "projeções e latentes de train/val/test", r"df\.to_csv\(res_p"),
    A("outputs/metrics/confusion/*.csv", "classification", "run", "Fase 5", EN,
      "matriz de confusão no test, por espaço e classificador (`src/evaluate.py::confusion`)", "previsões de `src/classification.py::run`", r"to_csv\(cm_p"),
    A("outputs/metrics/saturation.csv", "classification", "saturation", "Fase 5", EN,
      "menor d com accuracy de validação até 0.5 p.p. do máximo da família", "outputs/metrics/classification.csv", r"sat\.to_csv"),
    A("outputs/figures/classification_saturation.png", "classification", "figures", "Fase 5", EN,
      "accuracy de validação por d, família e classificador", "outputs/metrics/classification.csv"),
    A("outputs/figures/classification_confusions.png", "classification", "figures", "Fase 5", EN,
      "matrizes de confusão normalizadas de espaços representativos", "outputs/metrics/confusion/*.csv"),
    # ------------------------------------------------------------------ Fase 6
    A("outputs/features/*.npy", "retrieval", "compute_features", "Fase 6", EN,
      "silhueta, Sobel, HOG e LBP (`src/retrieval.py::feat_silhouette`, `src/retrieval.py::feat_sobel`, `src/retrieval.py::feat_hog`, `src/retrieval.py::feat_lbp`)", f"{TRAIN}, {VAL}, {TEST}", r"save_npy\(F, p"),
    A("outputs/retrieval/*query_idx_test.npy", "retrieval", "query_indices", "Fase 6", EN,
      "2000 queries estratificadas do test e 500 para a proxy visual", "outputs/data/y_test.npy", r"save_npy\(q"),
    A("outputs/retrieval/proxy_*.npy", "retrieval", "proxy_cache", "Fase 6", EN,
      "SSIM (`src/evaluate.py::SSIMGallery`) e IoU de silhueta (`src/evaluate.py::silhouette_iou`) de cada query contra a galeria", f"outputs/data/X_train_*.npy, outputs/data/X_test_*.npy", r"save_npy\([SI], p"),
    A("outputs/metrics/retrieval.csv", "retrieval", "evaluate_all", "Fase 6", EN,
      "P@k, consistency@k, mAP (`src/retrieval.py::label_metrics`) e proxy visual (`src/retrieval.py::proxy_metrics`), por espaço e distância", "espaços de `src/retrieval.py::_registry`, outputs/retrieval/*.npy", r"df\.to_csv\(res_p"),
    A("outputs/metrics/retrieval_vs_classification.csv", "retrieval", "compare_with_classification", "Fase 6", EN,
      "junta a accuracy de classificação às métricas de retrieval (euclidiana)", "outputs/metrics/retrieval.csv, outputs/metrics/classification.csv", r"r\.to_csv\(p"),
    A("outputs/metrics/retrieval_rank_agreement.csv", "retrieval", "compare_with_classification", "Fase 6", EN,
      "Spearman e Kendall entre rankings de espaços, com e sem d=2", "outputs/metrics/retrieval_vs_classification.csv", r"to_csv\(pc"),
    A("outputs/figures/retrieval_vs_classification.png", "retrieval", "compare_with_classification", "Fase 6", EN,
      "accuracy de classificação vs concordância com a proxy visual, por espaço", "outputs/metrics/retrieval_vs_classification.csv"),
    A("outputs/figures/retrieval_panel_q*.png", "retrieval", "panels", "Fase 6", EN,
      "query e top-8 em 12 espaços, com `src/retrieval.py::retrieve` (euclidiana)", "outputs/retrieval/query_idx_test.npy, espaços de `src/retrieval.py::_registry`"),
    # ------------------------------------------------------------------ Fase 8
    A("outputs/failures/failure_cases.csv", "failures", "main", "Fase 8", EN,
      "mineração automática (`src/failures.py::mine`) e causa provável (`src/failures.py::cause`)", "features de train/test dos 6 espaços", r"df\.to_csv\(p"),
    A("outputs/failures/failure_causes_summary.csv", "failures", "main", "Fase 8", EN,
      "nº de casos por tipo e causa", "outputs/failures/failure_cases.csv", r"summ\.to_csv"),
    A("outputs/failures/failures_shared_across_spaces.csv", "failures", "main", "Fase 8", EX,
      "imagens do test minadas em mais do que um espaço", "outputs/failures/failure_cases.csv", r"to_csv\(pm"),
    A("outputs/failures/failure_pairs_by_space.csv", "failures", "compare_spaces", "Fase 8", AD,
      "todos os erros 1-NN do test, por par de classes", "features de train/test dos 6 espaços", r"pairs\.to_csv"),
    A("outputs/failures/failure_overlap_jaccard.csv", "failures", "compare_spaces", "Fase 8", AD,
      "Jaccard dos conjuntos de imagens falhadas entre espaços", "features de train/test dos 6 espaços", r"J\.to_csv"),
    A("outputs/figures/failures_specialized_comparison.png", "failures", "compare_spaces", "Fase 8", AD,
      "pares mais trocados (ConvAE, VAE, DAE, d=32) e Jaccard", "outputs/failures/failure_pairs_by_space.csv, outputs/failures/failure_overlap_jaccard.csv"),
    A("outputs/figures/failures_*.png", "failures", "figure_by_cause", "Fase 8", EN,
      "pares minados, agrupados pela causa de `src/failures.py::cause`", f"outputs/failures/failure_cases.csv, {TRAIN}, {TEST}"),
    # ------------------------------------------------------------------ Fase 9
    A("outputs/metrics/provenance.csv", "provenance", "main", "Fase 9", EX,
      "regras de `src/provenance.py::tabela_artefactos`; a linha é procurada no código de cada vez (`src/provenance.py::_linha`)", "src/*.py, outputs/manifest.json", r'to_csv\(paths\["provenance"', "python -m src.provenance"),
    A("outputs/metrics/scripts.csv", "provenance", "main", "Fase 9", EX,
      "mapa de `src/provenance.py::tabela_scripts`; as dependências são lidas dos imports (`src/provenance.py::_dependencias`)", "src/*.py", r'to_csv\(paths\["scripts"', "python -m src.provenance"),
    A("outputs/metrics/requisitos.csv", "provenance", "main", "Fase 9", EX,
      "lista de `src/provenance.py::tabela_requisitos`; confirma que as funções e os artefactos citados existem", "src/*.py, outputs/", r'to_csv\(paths\["requisitos"', "python -m src.provenance"),
]

# ------------------------------------------------------------------------------- scripts
SCRIPTS = [
    ("config.yaml", EN, "Todas", "configuração central: seed, splits e hiperparâmetros de todas as fases", "", "lido por `src/utils.py::load_config`"),
    ("src/data.py", EN, "Fase 0", "leitura IDX, normalização [0,1], splits estratificados", "prepare, load_split, viz_indices, stratified_subset", CMD["src/data.py"]),
    ("src/representation.py", EN, "Fases 4–8", "funções obrigatórias: projeção/reconstrução PCA e encode/decode dos AEs", "pca_project, pca_reconstruct, ae_encode, ae_decode, ae_reconstruct", "importado (sem comando próprio)"),
    ("src/models.py", EN, "Fases 4–5", "DenseAE, ConvAE, TransformerAE, ConvVAE, CNN supervisionada", "DenseAE, ConvAE, ConvVAE, TransformerAE, SupervisedCNN, build_ae", "importado (sem comando próprio)"),
    ("src/train_ae.py", EN, "Teste de GPU, Fase 4", "script único de treino (--arch --dim), grelha, smoke test, sondagem de GPU, VAE e DAE", "fit, train_one, collect, grid, smoke, probe, train_loss, kl_div, add_noise", CMD["src/train_ae.py"]),
    ("src/retrieval.py", EN, "Fase 6", "retrieve(query_image, space, k, metric), registo de espaços, descritores, avaliação, painéis", "retrieve, available_spaces, _registry, compute_features, label_metrics, proxy_metrics, evaluate_all, panels", CMD["src/retrieval.py"]),
    ("src/evaluate.py", EN, "Todas", "kNN em GPU e todas as métricas", "knn_search, knn_predict, cls_metrics, confusion, logreg, embedding_quality, clustering_metrics, twonn, ssim_pairs, SSIMGallery, silhouette_iou", "importado (sem comando próprio)"),
    ("src/utils.py", EN, "Todas", "config, seeds, logs, manifesto, gravação atómica", "load_config, set_seed, get_logger, register, save_npy, save_json, out", "importado (sem comando próprio)"),
    ("notebooks/report.ipynb", EN, "Fase 9", "relatório final; só lê outputs/", "", "jupyter nbconvert --execute --inplace notebooks/report.ipynb"),
    ("README.md", EN, "Fase 9", "como correr, estrutura, seeds, nota sobre a extensão de cor", "", "-"),
    ("src/eda.py", "implementa o enunciado", "Fase 1", "imagens médias, distâncias entre classes, kNN em pixels", "class_distance_matrix, main", CMD["src/eda.py"]),
    ("src/pca_lda.py", "implementa o enunciado", "Fase 2", "PCA, whitening, por classe, IPCA, KPCA, LDA, TwoNN", "fit_pca, step_full, step_reconstruction, step_whitening, step_knn_curve, step_per_class, step_ipca, step_kpca, step_lda, step_twonn", CMD["src/pca_lda.py"]),
    ("src/viz_nonlinear.py", "implementa o enunciado", "Fase 3", "t-SNE e UMAP com T, C e preservação kNN", "inputs, run_tsne, run_umap, main, specialized_figures", CMD["src/viz_nonlinear.py"]),
    ("src/embeddings_viz.py", "implementa o enunciado", "Fase 4", "reconstruções, grelha UMAP, agrupamento e ranking", "reconstructions, embed2d, embeddings_and_clustering", CMD["src/embeddings_viz.py"]),
    ("src/train_cnn.py", "implementa o enunciado", "Fase 5", "CNN supervisionada (teto de referência)", "main, predict", CMD["src/train_cnn.py"]),
    ("src/classification.py", "implementa o enunciado", "Fase 5", "os mesmos classificadores em todos os espaços; saturação", "spaces, mlp_fit_predict, run, saturation, figures", CMD["src/classification.py"]),
    ("src/failures.py", "implementa o enunciado", "Fase 8", "mineração de falhas, causa provável, comparação entre espaços", "mine, cause, figure_by_cause, compare_spaces", CMD["src/failures.py"]),
    ("src/specialized.py", AD, "Adenda (4.4)", "amostras, interpolação, painéis e veredicto H1–H3", "summary_table, recon_panels, samples, interpolation, hypotheses", CMD["src/specialized.py"]),
    ("src/robustness.py", AD, "Adenda (4.4)", "queries ruidosas contra galeria limpa", "compute, reconstruct, figure", CMD["src/robustness.py"]),
    ("src/plots.py", "apoio", "Todas", "estilo das figuras e savefig com registo no manifesto", "savefig, scatter_classes, show_images, heatmap", "importado (sem comando próprio)"),
    ("src/provenance.py", "apoio", "Fase 9", "este mapa de artefactos, scripts e requisitos", "main", "python -m src.provenance"),
    ("run_all.sh", "apoio", "Todas", "corre o projeto inteiro por ordem", "", "./run_all.sh"),
]


# ----------------------------------------------------------------------------- requisitos
def Q(fase, requisito, codigo, artefactos, secao, nota=""):
    return dict(fase=fase, requisito=requisito, codigo=codigo, artefactos=artefactos, secao=secao, nota=nota)


REQUISITOS = [
    Q("Regras", "seed global 42 em python/numpy/torch", "`src/utils.py::set_seed`", "config.yaml", "Fase 0"),
    Q("Regras", "versões dos pacotes guardadas", "run_all.sh", "outputs/env.txt", "Fase 0"),
    Q("Regras", "DECISIONS, LOG, manifesto e um log por script", "`src/utils.py::register`, `src/utils.py::get_logger`", "docs/DECISIONS.md, docs/LOG.md, outputs/manifest.json, outputs/logs/*", "Mapa do código"),
    Q("Regras", "tudo retomável e idempotente", "`src/utils.py::exists`, `src/train_ae.py::fit` (last.pt)", "run_all.sh", "Mapa do código", "run_all.sh não foi corrido do zero numa pasta limpa; cada passo correu pelos mesmos comandos"),
    Q("Fase 0", "splits estratificados 54k/6k; test intocável", "`src/data.py::prepare`", "outputs/splits/*_idx.npy", "Fase 0"),
    Q("Fase 0", "normalização [0,1], versões flatten 784 e 1×28×28", "`src/data.py::prepare`, `src/data.py::load_split`", "outputs/data/*", "Fase 0"),
    Q("Fase 1", "imagem média por classe", "`src/eda.py::main`", "outputs/figures/eda_mean_images.png", "Fase 1"),
    Q("Fase 1", "distâncias intra vs inter-classe (10×10)", "`src/eda.py::class_distance_matrix`", "outputs/metrics/eda_class_distances.csv, outputs/figures/eda_class_distances.png", "Fase 1"),
    Q("Fase 1", "kNN em pixels, confusão e hipótese das classes de tronco", "`src/eda.py::main`, `src/evaluate.py::knn_predict`", "outputs/metrics/eda_knn_raw_confusion.csv, outputs/metrics/eda_confused_pairs.csv, outputs/metrics/eda_summary.json", "Fase 1"),
    Q("GPU", "nvidia-smi e plano registados no LOG", "`src/train_ae.py::probe`", "docs/LOG.md", "Fase 0 (0.2)"),
    Q("GPU", "smoke test: 1 época, d=2 e d=128", "`src/train_ae.py::smoke`", "outputs/metrics/smoke_test.csv", "Fase 0 (0.2)"),
    Q("GPU", "sondagem de memória até OOM", "`src/train_ae.py::probe`", "outputs/metrics/gpu_probe.csv", "Fase 0 (0.2)", "ficou batch 256, não o maior que cabe, porque batches maiores prejudicam a convergência (`batch_convergence.csv`)"),
    Q("GPU", "dados em GPU, sem gargalo do loader", "`src/train_ae.py::GPUBatchLoader`, `src/train_ae.py::load_gpu_data`", "outputs/ae/*/metrics.json", "Fase 0 (0.2)"),
    Q("Fase 2", "PCA completo; componentes para 80/90/95/99%", "`src/pca_lda.py::step_full`", "outputs/pca/pca_full.joblib, outputs/metrics/pca_components_thresholds.csv", "Fase 2"),
    Q("Fase 2", "MSE para k ∈ {2,...,128}, painéis, eigen-fashion", "`src/pca_lda.py::step_reconstruction`", "outputs/metrics/pca_reconstruction.csv, outputs/figures/pca_reconstructions.png, outputs/figures/pca_eigenfashion.png", "Fase 2"),
    Q("Fase 2", "PCA com e sem whitening: projeções, modelos e kNN", "`src/pca_lda.py::step_whitening`", "outputs/pca/pca_k*.joblib, outputs/pca/proj/pca_k*.npy, outputs/metrics/pca_knn_whitening.csv", "Fase 2"),
    Q("Fase 2", "funções de projeção e reconstrução PCA", "`src/representation.py::pca_project`, `src/representation.py::pca_reconstruct`", "outputs/pca/pca_full.joblib", "Fase 2"),
    Q("Fase 2", "PCA por classe vs global", "`src/pca_lda.py::step_per_class`", "outputs/metrics/pca_per_class.csv, outputs/figures/pca_per_class.png", "fora do relatório", "calculado e guardado, mas retirado do relatório por opção"),
    Q("Fase 2", "IncrementalPCA e KernelPCA RBF (cerca de 10k)", "`src/pca_lda.py::step_ipca`, `src/pca_lda.py::step_kpca`", "outputs/pca/ipca_k128.joblib, outputs/pca/kpca_rbf_k128.joblib, outputs/metrics/ipca_vs_pca.csv, outputs/metrics/kpca_knn.csv", "Fase 2"),
    Q("Fase 2", "LDA até 9 componentes; comparação com PCA em kNN/LogReg", "`src/pca_lda.py::step_lda`", "outputs/lda/lda.joblib, outputs/lda/lda_*.npy, outputs/metrics/lda_vs_pca.csv", "Fase 2"),
    Q("Fase 2", "TwoNN global e por classe vs k de saturação", "`src/evaluate.py::twonn`, `src/pca_lda.py::step_twonn`", "outputs/metrics/twonn.csv, outputs/figures/twonn.png", "Fase 2"),
    Q("Fase 3", "t-SNE e UMAP com varrimento, em 10k pontos", "`src/viz_nonlinear.py::run_tsne`, `src/viz_nonlinear.py::run_umap`, `src/viz_nonlinear.py::main`", "outputs/viz/tsne_*.npy, outputs/viz/umap_*.npy, outputs/figures/viz_raw.png, outputs/figures/viz_pca50.png, outputs/figures/viz_conv_ae_d32.png", "Fase 3"),
    Q("Fase 3", "trustworthiness, continuity, preservação kNN", "`src/evaluate.py::embedding_quality`", "outputs/metrics/viz_quality.csv", "Fase 3"),
    Q("Fase 4", "DenseAE, ConvAE e TransformerAE, sem skip connections", "`src/models.py::DenseAE`, `src/models.py::ConvAE`, `src/models.py::TransformerAE`", "outputs/ae/*/best.pt", "Fase 4"),
    Q("Fase 4", "18 modelos, script único --arch --dim, mesmo protocolo", "`src/train_ae.py::fit`, `src/train_ae.py::train_one`, `src/train_ae.py::grid`", "outputs/ae/*/history.csv, outputs/latents/*.npy, outputs/metrics/ae_summary.csv", "Fase 4"),
    Q("Fase 4", "funções de encode e decode", "`src/representation.py::ae_encode`, `src/representation.py::ae_decode`", "outputs/ae/*/best.pt", "Fase 4"),
    Q("Fase 4", "MSE/SSIM, parâmetros, tempo, pico de memória", "`src/train_ae.py::recon_metrics`, `src/train_ae.py::train_one`", "outputs/ae/*/metrics.json", "Fase 4"),
    Q("Fase 4", "painel original vs PCA vs cada AE", "`src/embeddings_viz.py::reconstructions`", "outputs/figures/recon_panel_d*.png, outputs/figures/recon_panel_by_d.png", "Fase 4"),
    Q("Fase 4", "grelha UMAP de todos os modelos, com PCA e LDA; d=2 direto", "`src/embeddings_viz.py::embed2d`, `src/embeddings_viz.py::embeddings_and_clustering`", "outputs/viz/emb_*.npy, outputs/figures/embeddings_grid.png", "Fase 4"),
    Q("Fase 4", "silhouette, DB, kNN, NMI/ARI e ranking", "`src/evaluate.py::clustering_metrics`, `src/embeddings_viz.py::embeddings_and_clustering`", "outputs/metrics/embedding_clustering.csv, outputs/figures/embedding_ranking.png", "Fase 4"),
    Q("Fase 5", "mesmos classificadores em todos os espaços", "`src/classification.py::spaces`, `src/classification.py::run`", "outputs/metrics/classification.csv", "Fase 5"),
    Q("Fase 5", "matrizes de confusão", "`src/classification.py::run`, `src/classification.py::figures`", "outputs/metrics/confusion/*.csv, outputs/figures/classification_confusions.png", "Fase 5"),
    Q("Fase 5", "curva de saturação", "`src/classification.py::saturation`", "outputs/metrics/saturation.csv, outputs/figures/classification_saturation.png", "Fase 5"),
    Q("Fase 5", "teto: CNN supervisionada com pesos guardados", "`src/models.py::SupervisedCNN`, `src/train_cnn.py::main`", "outputs/cnn/best.pt, outputs/cnn/metrics.json", "Fase 5"),
    Q("Fase 6", "retrieve(query_image, space, k, metric) com registo central", "`src/retrieval.py::retrieve`, `src/retrieval.py::_registry`, `src/retrieval.py::available_spaces`", "outputs/metrics/retrieval.csv", "Fase 6"),
    Q("Fase 6", "perspetivas shape, texture/edges e learned", "`src/retrieval.py::feat_silhouette`, `src/retrieval.py::feat_sobel`, `src/retrieval.py::feat_hog`, `src/retrieval.py::feat_lbp`", "outputs/features/*.npy", "Fase 6"),
    Q("Fase 6", "euclidiana e cosseno; kNN em GPU com torch.cdist em blocos", "`src/evaluate.py::pairwise_dist`, `src/evaluate.py::knn_search`", "outputs/metrics/retrieval.csv", "Fase 6"),
    Q("Fase 6", "precision@k, consistency@k, mAP para k ∈ {1,5,10,20}", "`src/retrieval.py::label_metrics`", "outputs/metrics/retrieval.csv", "Fase 6"),
    Q("Fase 6", "proxy visual (SSIM/IoU), Spearman e NDCG vs classificação", "`src/retrieval.py::proxy_cache`, `src/retrieval.py::proxy_metrics`, `src/retrieval.py::compare_with_classification`", "outputs/retrieval/proxy_*.npy, outputs/metrics/retrieval_vs_classification.csv, outputs/metrics/retrieval_rank_agreement.csv", "Fase 6"),
    Q("Fase 6", "painéis query + top-k", "`src/retrieval.py::panels`", "outputs/figures/retrieval_panel_q*.png", "Fase 6"),
    Q("Fase 8", "pelo menos 5 casos minados, em vários espaços", "`src/failures.py::mine`", "outputs/failures/failure_cases.csv", "Fase 8"),
    Q("Fase 8", "figuras agrupadas por causa provável", "`src/failures.py::cause`, `src/failures.py::figure_by_cause`", "outputs/figures/failures_*.png", "Fase 8"),
    Q("Fase 9", "relatório que só lê outputs/ e README", "notebooks/report.ipynb", "notebooks/report.ipynb, README.md", "todas"),
    Q("Adenda A", "`vae_conv`: mu+logvar, reparametrização, MSE somado + β·KL, warm-up", "`src/models.py::ConvVAE`, `src/train_ae.py::kl_div`, `src/train_ae.py::train_loss`", "outputs/ae/vae_conv_d*/best.pt, outputs/latents/*_logvar_*.npy", "4.4"),
    Q("Adenda A", "KL, MSE e unidades ativas", "`src/train_ae.py::vae_stats`, `src/train_ae.py::vae_val`", "outputs/ae/vae_conv_d*/metrics.json, outputs/metrics/specialized_summary.csv", "4.4"),
    Q("Adenda A", "amostras do prior 10×10 e interpolação vs PCA", "`src/specialized.py::samples`, `src/specialized.py::interpolation`", "outputs/figures/vae_samples.png, outputs/figures/interpolation_d*.png, outputs/metrics/interpolation.csv", "4.4"),
    Q("Adenda A", "`dae_conv`: σ=0.3, clip, alvo limpo, avaliação limpa", "`src/train_ae.py::add_noise`, `src/train_ae.py::train_loss`, `src/train_ae.py::validate`", "outputs/ae/dae_conv_d*/best.pt", "4.4"),
    Q("Adenda A", "robustez σ ∈ {0, 0.1, 0.3, 0.5}", "`src/robustness.py::compute`", "outputs/metrics/robustness.csv, outputs/figures/robustness.png", "4.4"),
    Q("Adenda B", "especializados em todas as fases", "`src/retrieval.py::_registry`, `src/classification.py::spaces`, `src/embeddings_viz.py::embeddings_and_clustering`, `src/viz_nonlinear.py::specialized_figures`, `src/failures.py::compare_spaces`", "outputs/figures/viz_specialized.png, outputs/failures/failure_pairs_by_space.csv", "Fases 3–8"),
    Q("Adenda C", "H1–H3 antes dos resultados, com veredicto", "`src/specialized.py::hypotheses`", "docs/DECISIONS.md, outputs/metrics/hypotheses.csv", "4.4"),
    Q("Proibido", "Fase 7 (cor), UMAP supervisionado, U-Net, fusão de espaços, hubness, queries fora do dataset", "-", "", "-", "não feito, como pedido; o VAE e o DAE só entram por causa da adenda"),
]


# ------------------------------------------------------------------------------ verificação
def _defs(script: str) -> dict[str, ast.AST]:
    tree = ast.parse((ROOT / script).read_text())
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _linha(rule: dict) -> int:
    """Linha da escrita do artefacto, dentro da função indicada (a última que corresponde)."""
    node = _defs(rule["script"]).get(rule["funcao"])
    if node is None:
        raise ValueError(f"{rule['script']}: a função {rule['funcao']} não existe")
    base = Path(rule["padrao"]).name.split("*")[0]
    w = rule["w"] or re.escape(base)
    lines = (ROOT / rule["script"]).read_text().split("\n")
    hits = [i for i in range(node.lineno, node.end_lineno + 1) if re.search(w, lines[i - 1])]
    if not hits:
        raise ValueError(f"{rule['padrao']}: nada corresponde a {w!r} em {rule['script']}::{rule['funcao']}")
    return hits[-1]


def _citacoes(txt: str) -> None:
    for script, fn in re.findall(r"(src/\w+\.py)::(\w+)", txt):
        if fn not in _defs(script):
            raise ValueError(f"citação inválida: {script}::{fn}")


def _regra(rel: str) -> dict | None:
    return next((r for r in ARTEFACTOS if fnmatch(rel, r["padrao"])), None)


def _ficheiros() -> list[str]:
    fora = {"outputs/manifest.json", "outputs/env.txt"}
    proprios = {"outputs/metrics/provenance.csv", "outputs/metrics/scripts.csv", "outputs/metrics/requisitos.csv"}
    fs = [str(p.relative_to(ROOT)) for p in OUT.rglob("*") if p.is_file() and "logs" not in p.parts and p.suffix != ".lock"]
    return sorted((set(fs) | proprios) - fora)


def _existe(padrao: str) -> bool:
    return bool(list(ROOT.glob(padrao)))


def _dependencias() -> dict[str, list[str]]:
    """Script -> scripts de src/ que o importam (lido dos imports)."""
    mods = [p.stem for p in SRC.glob("*.py")]
    usado = {m: set() for m in mods}
    for p in SRC.glob("*.py"):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module in usado and n.module != p.stem:
                usado[n.module].add(p.stem)
    return {m: sorted(v) for m, v in usado.items()}


# -------------------------------------------------------------------------------- saídas
def tabela_artefactos() -> pd.DataFrame:
    man = load_json(OUT / "manifest.json")["artifacts"]
    rows, sem_origem = [], []
    usadas = set()
    for rel in _ficheiros():
        r = _regra(rel)
        if r is None:
            sem_origem.append(rel)
            continue
        usadas.add(r["padrao"])
        rows.append({"artefacto": rel, "descricao": man.get(rel, {}).get("descricao", ""), "fase": r["fase"], "origem": r["origem"],
                     "script": r["script"], "funcao": r["funcao"], "linha": _linha(r),
                     "comando": r["cmd"] or CMD[r["script"]], "calculo": r["calculo"], "dados": r["dados"]})
    if sem_origem:
        raise ValueError(f"artefactos sem origem registada: {sem_origem}")
    vazias = [r["padrao"] for r in ARTEFACTOS if r["padrao"] not in usadas]
    if vazias:
        raise ValueError(f"regras sem ficheiros: {vazias}")
    for r in ARTEFACTOS:
        _citacoes(r["calculo"] + " " + r["dados"])
    return pd.DataFrame(rows)


def tabela_scripts() -> pd.DataFrame:
    citados = {s[0] for s in SCRIPTS}
    falta = [f"src/{p.name}" for p in SRC.glob("*.py") if f"src/{p.name}" not in citados and p.name != "__init__.py"]
    if falta:
        raise ValueError(f"scripts fora do mapa: {falta}")
    dep = _dependencias()
    rows = []
    for path, papel, fase, descr, fns, cmd in SCRIPTS:
        if not (ROOT / path).exists():
            raise ValueError(f"não existe: {path}")
        fl = [f.strip() for f in fns.split(",") if f.strip()]
        for f in fl:
            if f not in _defs(path):
                raise ValueError(f"{path}: a função {f} não existe")
        _citacoes(cmd)
        stem = Path(path).stem
        rows.append({"script": path, "papel": papel, "fase": fase, "descricao": descr, "funcoes_principais": ", ".join(fl),
                     "comando": cmd, "importado_por": ", ".join(dep.get(stem, [])) if path.startswith("src/") else "",
                     "linhas": sum(1 for _ in open(ROOT / path, "rb")) if not path.endswith(".ipynb") else None})
    return pd.DataFrame(rows)


def tabela_requisitos() -> pd.DataFrame:
    rows = []
    for q in REQUISITOS:
        _citacoes(q["codigo"])
        pads = [p.strip() for p in q["artefactos"].split(",") if p.strip()]
        for p in pads:
            if not _existe(p):
                raise ValueError(f"{q['requisito']}: não existe {p}")
        rows.append(q)
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    def cell(v):
        return "" if pd.isna(v) else str(v).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False))


def documento(art: pd.DataFrame, scr: pd.DataFrame, req: pd.DataFrame) -> str:
    def loc(r):
        return f"`{r.script}`, `{r.funcao}()`, linha {r.linha}"
    a = art.assign(onde=art.apply(loc, axis=1))
    fig = a[a.artefacto.str.startswith("outputs/figures/")]
    tab = a[a.artefacto.str.contains(r"outputs/(?:metrics|failures)/") & ~a.artefacto.str.contains("metrics/confusion/")]
    # agrupados por regra para não repetir centenas de linhas iguais
    resto = a[~a.index.isin(fig.index) & ~a.index.isin(tab.index)].copy()
    resto["regra"] = resto.artefacto.map(lambda f: _regra(f)["padrao"])
    grupos = resto.groupby("regra", sort=False).agg(n=("artefacto", "size"), onde=("onde", "first"), comando=("comando", "first"),
                                                    calculo=("calculo", "first")).reset_index()
    cols = ["artefacto", "fase", "onde", "comando", "calculo", "dados"]
    return "\n".join([
        "# Proveniência: de onde vem cada resultado",
        "",
        "Gerado por `python -m src.provenance`. As linhas são procuradas no código de cada vez, e o script falha",
        "se um ficheiro de `outputs/` não tiver origem ou se uma função citada não existir.",
        "",
        f"## Scripts ({len(scr)})",
        "",
        _md(scr),
        "",
        f"## Requisitos do enunciado e da adenda ({len(req)})",
        "",
        _md(req),
        "",
        f"## Figuras ({len(fig)})",
        "",
        _md(fig[cols]),
        "",
        f"## Tabelas ({len(tab)})",
        "",
        _md(tab[cols]),
        "",
        f"## Modelos, projeções, latentes e restantes ficheiros ({len(resto)}, agrupados)",
        "",
        _md(grupos),
        "",
    ])


def main():
    paths = {"provenance": out("metrics", "provenance.csv"), "scripts": out("metrics", "scripts.csv"),
             "requisitos": out("metrics", "requisitos.csv")}
    register(paths["provenance"], "origem de cada artefacto: script, função, linha, comando, cálculo e dados de entrada")
    register(paths["scripts"], "mapa dos scripts: papel, fase, funções principais, comando, quem os importa")
    register(paths["requisitos"], "requisitos do enunciado e da adenda: código, artefactos e secção do relatório")
    art, scr, req = tabela_artefactos(), tabela_scripts(), tabela_requisitos()
    art.to_csv(paths["provenance"], index=False)
    scr.to_csv(paths["scripts"], index=False)
    req.to_csv(paths["requisitos"], index=False)
    (ROOT / "docs" / "PROVENIENCIA.md").write_text(documento(art, scr, req))
    log.info(f"{len(art)} artefactos, {len(scr)} scripts, {len(req)} requisitos verificados")


if __name__ == "__main__":
    main()
