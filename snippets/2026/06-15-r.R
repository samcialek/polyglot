# Principal Component Analysis from scratch using eigendecomposition.

pca <- function(X, n_components = NULL) {
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(n_components)) n_components <- min(n, p)

  # Center the data
  means <- colMeans(X)
  X_centered <- sweep(X, 2, means)

  # Covariance matrix
  cov_matrix <- crossprod(X_centered) / (n - 1)

  # Eigendecomposition
  eig <- eigen(cov_matrix, symmetric = TRUE)
  values <- eig$values[1:n_components]
  vectors <- eig$vectors[, 1:n_components, drop = FALSE]

  # Project data
  scores <- X_centered %*% vectors

  total_var <- sum(eig$values)
  list(
    scores = scores,
    loadings = vectors,
    eigenvalues = values,
    variance_explained = values / total_var,
    cumulative_var = cumsum(values) / total_var,
    center = means
  )
}

# Demo: iris-like simulated data
set.seed(42)
n <- 150
# 3 clusters with correlated features
mu <- list(c(5, 3.5, 1.4, 0.2), c(6, 2.8, 4.5, 1.3), c(6.5, 3, 5.5, 2))
X <- do.call(rbind, lapply(1:3, function(k) {
  MASS::mvrnorm(n/3, mu[[k]], diag(4) * 0.3)
}))
colnames(X) <- paste0("V", 1:4)

result <- pca(X, n_components = 4)

cat("=== PCA Results ===
")
cat("
Variance explained:
")
for (i in 1:4) {
  bar <- paste(rep("#", round(result$variance_explained[i] * 40)), collapse = "")
  cat(sprintf("  PC%d: %5.1f%% %s (cumulative: %.1f%%)
",
              i, result$variance_explained[i] * 100, bar,
              result$cumulative_var[i] * 100))
}

cat("
Loadings (first 2 PCs):
")
cat(sprintf("  %-4s  %7s  %7s
", "Var", "PC1", "PC2"))
for (j in 1:4) {
  cat(sprintf("  V%-3d  %7.3f  %7.3f
", j, result$loadings[j, 1], result$loadings[j, 2]))
}
