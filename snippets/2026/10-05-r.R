# Ridge regression with k-fold cross-validation for lambda selection.

ridge_regression <- function(X, y, lambda) {
  # Closed-form: beta = (X'X + lambda*I)^{-1} X'y
  p <- ncol(X)
  XtX <- crossprod(X)
  Xty <- crossprod(X, y)
  beta <- solve(XtX + lambda * diag(p), Xty)
  as.vector(beta)
}

cv_ridge <- function(X, y, lambdas, k = 5) {
  n <- nrow(X)
  folds <- sample(rep(1:k, length.out = n))

  cv_errors <- sapply(lambdas, function(lam) {
    fold_errors <- sapply(1:k, function(fold) {
      train <- folds != fold
      test <- folds == fold
      beta <- ridge_regression(X[train, , drop = FALSE], y[train], lam)
      preds <- X[test, , drop = FALSE] %*% beta
      mean((y[test] - preds)^2)
    })
    mean(fold_errors)
  })

  list(
    lambdas = lambdas,
    cv_errors = cv_errors,
    best_lambda = lambdas[which.min(cv_errors)],
    min_error = min(cv_errors)
  )
}

# Demo: correlated predictors (where ridge excels)
set.seed(42)
n <- 200; p <- 10
Sigma <- outer(1:p, 1:p, function(i, j) 0.7^abs(i - j))
X <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
true_beta <- c(3, -2, 0, 0, 1.5, 0, 0, -1, 0, 0.5)
y <- X %*% true_beta + rnorm(n, 0, 2)

# Cross-validate
lambdas <- 10^seq(-2, 3, length.out = 50)
cv_result <- cv_ridge(X, y, lambdas)

cat("=== Ridge Regression with CV ===
")
cat(sprintf("  Best lambda: %.4f
", cv_result$best_lambda))
cat(sprintf("  CV MSE:      %.4f
", cv_result$min_error))

# Compare OLS vs Ridge
ols_beta <- solve(crossprod(X), crossprod(X, y))
ridge_beta <- ridge_regression(X, y, cv_result$best_lambda)

cat("
  Coefficient comparison:
")
cat(sprintf("  %-6s %8s %8s %8s
", "Var", "True", "OLS", "Ridge"))
for (j in 1:p) {
  cat(sprintf("  X%-5d %8.2f %8.2f %8.2f
", j, true_beta[j], ols_beta[j], ridge_beta[j]))
}
