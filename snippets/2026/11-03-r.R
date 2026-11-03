# Bootstrap confidence intervals — nonparametric inference.

bootstrap_ci <- function(x, stat_fn = mean, n_boot = 10000, alpha = 0.05) {
  boot_stats <- replicate(n_boot, {
    sample_idx <- sample(seq_along(x), replace = TRUE)
    stat_fn(x[sample_idx])
  })

  list(
    estimate = stat_fn(x),
    ci_lower = quantile(boot_stats, alpha / 2),
    ci_upper = quantile(boot_stats, 1 - alpha / 2),
    se = sd(boot_stats),
    bias = mean(boot_stats) - stat_fn(x)
  )
}

# Demo: confidence interval for median income
set.seed(123)
incomes <- c(rlnorm(200, meanlog = 10.5, sdlog = 0.8))

cat("=== Bootstrap CI for Median Income ===
")
result <- bootstrap_ci(incomes, stat_fn = median)
cat(sprintf("  Median:  $%.0f
", result$estimate))
cat(sprintf("  95%% CI:  [$%.0f, $%.0f]
", result$ci_lower, result$ci_upper))
cat(sprintf("  SE:      $%.0f
", result$se))
cat(sprintf("  Bias:    $%.2f
", result$bias))

# Compare mean vs trimmed mean
cat("
=== Mean vs Trimmed Mean ===
")
ci_mean <- bootstrap_ci(incomes, stat_fn = mean)
ci_trim <- bootstrap_ci(incomes, stat_fn = function(x) mean(x, trim = 0.1))
cat(sprintf("  Mean:         $%.0f [%.0f, %.0f]
", ci_mean$estimate, ci_mean$ci_lower, ci_mean$ci_upper))
cat(sprintf("  Trimmed Mean: $%.0f [%.0f, %.0f]
", ci_trim$estimate, ci_trim$ci_lower, ci_trim$ci_upper))
