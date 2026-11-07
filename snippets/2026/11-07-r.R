# Simplified mixed-effects model estimation via EM algorithm.
# Y_ij = beta0 + beta1*x_ij + b_i + epsilon_ij

simulate_mixed_data <- function(n_groups = 20, obs_per_group = 10) {
  set.seed(42)
  beta0 <- 5.0      # fixed intercept
  beta1 <- 2.0      # fixed slope
  sigma_b <- 1.5    # random effect SD
  sigma_e <- 1.0    # residual SD

  group <- rep(1:n_groups, each = obs_per_group)
  b <- rnorm(n_groups, 0, sigma_b)  # random intercepts
  x <- rnorm(n_groups * obs_per_group, 0, 1)
  y <- beta0 + beta1 * x + b[group] + rnorm(length(x), 0, sigma_e)

  data.frame(y = y, x = x, group = group)
}

estimate_mixed_model <- function(df, max_iter = 50, tol = 1e-6) {
  # Simple EM for random intercept model
  groups <- unique(df$group)
  n_g <- length(groups)

  # Initial estimates
  sigma2_b <- 1.0
  sigma2_e <- 1.0

  for (iter in seq_len(max_iter)) {
    # E-step: estimate random effects
    b_hat <- numeric(n_g)
    for (i in seq_along(groups)) {
      idx <- df$group == groups[i]
      n_i <- sum(idx)
      resid_i <- df$y[idx] - mean(df$y)  # simplified
      b_hat[i] <- (sigma2_b / (sigma2_b + sigma2_e / n_i)) * mean(resid_i)
    }

    # M-step: update fixed effects via OLS on residuals
    df$y_adj <- df$y - b_hat[df$group]
    fit <- lm(y_adj ~ x, data = df)

    # Update variance components
    resid <- df$y - predict(fit, df) - b_hat[df$group]
    sigma2_e_new <- mean(resid^2)
    sigma2_b_new <- max(var(b_hat) - sigma2_e / mean(table(df$group)), 0.01)

    if (abs(sigma2_e_new - sigma2_e) + abs(sigma2_b_new - sigma2_b) < tol) break
    sigma2_e <- sigma2_e_new
    sigma2_b <- sigma2_b_new
  }

  list(
    beta = coef(fit),
    sigma_b = sqrt(sigma2_b),
    sigma_e = sqrt(sigma2_e),
    random_effects = b_hat,
    iterations = iter
  )
}

# Run
df <- simulate_mixed_data()
result <- estimate_mixed_model(df)

cat("=== Mixed Effects Model (EM) ===
")
cat(sprintf("  beta0 (true=5.0): %.2f
", result$beta[1]))
cat(sprintf("  beta1 (true=2.0): %.2f
", result$beta[2]))
cat(sprintf("  sigma_b (true=1.5): %.2f
", result$sigma_b))
cat(sprintf("  sigma_e (true=1.0): %.2f
", result$sigma_e))
cat(sprintf("  Converged in %d iterations
", result$iterations))
