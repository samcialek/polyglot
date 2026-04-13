# Metropolis-Hastings MCMC sampler for Bayesian posterior estimation.

metropolis_hastings <- function(log_target, init, n_iter = 10000,
                                 proposal_sd = 1, burn_in = 1000) {
  samples <- numeric(n_iter)
  current <- init
  accepted <- 0

  for (i in seq_len(n_iter)) {
    proposal <- rnorm(1, current, proposal_sd)
    log_ratio <- log_target(proposal) - log_target(current)

    if (log(runif(1)) < log_ratio) {
      current <- proposal
      accepted <- accepted + 1
    }
    samples[i] <- current
  }

  list(
    samples = samples[(burn_in + 1):n_iter],
    acceptance_rate = accepted / n_iter
  )
}

# Demo: Estimate posterior of normal mean with known variance
# Prior: mu ~ N(0, 10^2), Likelihood: x ~ N(mu, 2^2)
set.seed(42)
true_mu <- 3.5
data <- rnorm(50, mean = true_mu, sd = 2)

log_posterior <- function(mu) {
  log_prior <- dnorm(mu, mean = 0, sd = 10, log = TRUE)
  log_likelihood <- sum(dnorm(data, mean = mu, sd = 2, log = TRUE))
  log_prior + log_likelihood
}

result <- metropolis_hastings(log_posterior, init = 0, n_iter = 20000,
                               proposal_sd = 0.5, burn_in = 2000)

cat("=== MCMC Posterior Estimation ===
")
cat(sprintf("  True mu:         %.2f
", true_mu))
cat(sprintf("  Posterior mean:  %.2f
", mean(result$samples)))
cat(sprintf("  Posterior SD:    %.2f
", sd(result$samples)))
cat(sprintf("  95%% CI:          [%.2f, %.2f]
",
            quantile(result$samples, 0.025), quantile(result$samples, 0.975)))
cat(sprintf("  Acceptance rate: %.1f%%
", result$acceptance_rate * 100))
cat(sprintf("  MLE (x-bar):     %.2f
", mean(data)))
