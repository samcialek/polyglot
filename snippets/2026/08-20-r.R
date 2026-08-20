# Bayesian A/B testing with Beta-Binomial conjugate model.

bayesian_ab_test <- function(successes_a, trials_a, successes_b, trials_b,
                              prior_alpha = 1, prior_beta = 1, n_sim = 100000) {
  # Posterior distributions (Beta-Binomial conjugate)
  post_a <- rbeta(n_sim, prior_alpha + successes_a, prior_beta + trials_a - successes_a)
  post_b <- rbeta(n_sim, prior_alpha + successes_b, prior_beta + trials_b - successes_b)

  # P(B > A)
  prob_b_better <- mean(post_b > post_a)

  # Lift distribution
  lift <- (post_b - post_a) / post_a

  list(
    prob_b_better = prob_b_better,
    expected_rate_a = mean(post_a),
    expected_rate_b = mean(post_b),
    expected_lift = mean(lift),
    lift_ci = quantile(lift, c(0.025, 0.975)),
    risk_of_choosing_b = mean(pmax(post_a - post_b, 0))  # expected loss
  )
}

# Demo: website conversion test
set.seed(42)
# Control: 120 conversions out of 1000
# Variant: 145 conversions out of 1000
result <- bayesian_ab_test(120, 1000, 145, 1000)

cat("=== Bayesian A/B Test Results ===
")
cat(sprintf("  P(B > A):        %.1f%%
", result$prob_b_better * 100))
cat(sprintf("  Rate A:          %.2f%%
", result$expected_rate_a * 100))
cat(sprintf("  Rate B:          %.2f%%
", result$expected_rate_b * 100))
cat(sprintf("  Expected Lift:   %.1f%%
", result$expected_lift * 100))
cat(sprintf("  Lift 95%% CI:     [%.1f%%, %.1f%%]
",
            result$lift_ci[1] * 100, result$lift_ci[2] * 100))
cat(sprintf("  Risk (choose B): %.4f
", result$risk_of_choosing_b))
