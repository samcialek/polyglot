# Kaplan-Meier survival curve estimation from scratch.

kaplan_meier <- function(time, event) {
  # Sort by time
  ord <- order(time)
  time <- time[ord]
  event <- event[ord]

  unique_times <- sort(unique(time[event == 1]))
  n <- length(time)

  surv_prob <- 1.0
  results <- data.frame(
    time = numeric(0), n_risk = integer(0),
    n_event = integer(0), survival = numeric(0),
    se = numeric(0)
  )

  var_sum <- 0  # for Greenwood's formula
  for (t in unique_times) {
    n_risk <- sum(time >= t)
    n_event <- sum(time == t & event == 1)
    hazard <- n_event / n_risk
    surv_prob <- surv_prob * (1 - hazard)

    if (n_event > 0) {
      var_sum <- var_sum + n_event / (n_risk * (n_risk - n_event))
    }
    se <- surv_prob * sqrt(var_sum)

    results <- rbind(results, data.frame(
      time = t, n_risk = n_risk, n_event = n_event,
      survival = surv_prob, se = se
    ))
  }

  results$ci_lower <- pmax(results$survival - 1.96 * results$se, 0)
  results$ci_upper <- pmin(results$survival + 1.96 * results$se, 1)
  results
}

# Simulate clinical trial survival data
set.seed(42)
n <- 100
time <- rexp(n, rate = 0.1)              # true survival times
censor_time <- runif(n, 0, 30)           # censoring times
observed_time <- pmin(time, censor_time)  # what we observe
event <- as.integer(time <= censor_time)  # 1 = event, 0 = censored

km <- kaplan_meier(observed_time, event)

cat("=== Kaplan-Meier Survival Estimates ===
")
cat(sprintf("  Events: %d / %d (%.0f%% censored)
",
            sum(event), n, (1 - mean(event)) * 100))
cat("
  Time    At Risk  Events  Survival  95% CI
")
cat("  ", strrep("-", 55), "
", sep = "")
for (i in seq_len(min(nrow(km), 12))) {
  cat(sprintf("  %5.1f   %4d     %4d    %.3f     [%.3f, %.3f]
",
              km$time[i], km$n_risk[i], km$n_event[i],
              km$survival[i], km$ci_lower[i], km$ci_upper[i]))
}

# Median survival time
median_idx <- which(km$survival <= 0.5)[1]
if (!is.na(median_idx)) {
  cat(sprintf("
  Median survival time: %.1f
", km$time[median_idx]))
}
