# Propensity score matching for causal inference — observational data.

simulate_treatment_data <- function(n = 500) {
  set.seed(42)
  age <- rnorm(n, 45, 12)
  income <- rnorm(n, 60000, 15000)

  # Treatment assignment depends on covariates (confounding)
  logit_p <- -2 + 0.03 * age + 0.00002 * income
  prob_treat <- plogis(logit_p)
  treatment <- rbinom(n, 1, prob_treat)

  # Outcome depends on treatment AND covariates
  outcome <- 100 + 5 * treatment + 0.5 * age + 0.0003 * income + rnorm(n, 0, 10)

  data.frame(age, income, treatment, outcome)
}

propensity_score_match <- function(df) {
  # Estimate propensity scores via logistic regression
  ps_model <- glm(treatment ~ age + income, data = df, family = binomial)
  df$pscore <- predict(ps_model, type = "response")

  # Naive estimate (biased)
  naive <- mean(df$outcome[df$treatment == 1]) - mean(df$outcome[df$treatment == 0])

  # IPW estimate (inverse propensity weighting)
  w1 <- df$treatment / df$pscore
  w0 <- (1 - df$treatment) / (1 - df$pscore)
  ipw_ate <- weighted.mean(df$outcome, w1) - weighted.mean(df$outcome, w0)

  # Regression adjustment
  reg_model <- lm(outcome ~ treatment + age + income, data = df)
  reg_ate <- coef(reg_model)["treatment"]

  list(
    naive_ate = naive,
    ipw_ate = ipw_ate,
    regression_ate = reg_ate,
    true_ate = 5  # we know this from simulation
  )
}

# Run
df <- simulate_treatment_data()
results <- propensity_score_match(df)

cat("=== Causal Inference: Average Treatment Effect ===
")
cat(sprintf("  True ATE:       %.2f
", results$true_ate))
cat(sprintf("  Naive estimate: %.2f (biased — ignores confounders)
", results$naive_ate))
cat(sprintf("  IPW estimate:   %.2f
", results$ipw_ate))
cat(sprintf("  Regression adj: %.2f
", results$regression_ate))
