# Classical time series decomposition: trend + seasonal + residual.

decompose_ts <- function(x, period) {
  n <- length(x)

  # Trend: centered moving average
  trend <- rep(NA, n)
  half <- period %/% 2
  for (i in (half + 1):(n - half)) {
    if (period %% 2 == 0) {
      trend[i] <- (0.5 * x[i - half] + sum(x[(i - half + 1):(i + half - 1)]) + 0.5 * x[i + half]) / period
    } else {
      trend[i] <- mean(x[(i - half):(i + half)])
    }
  }

  # Seasonal: average deviation from trend by position in cycle
  detrended <- x - trend
  seasonal <- numeric(n)
  for (s in 1:period) {
    idx <- seq(s, n, by = period)
    seasonal_mean <- mean(detrended[idx], na.rm = TRUE)
    seasonal[idx] <- seasonal_mean
  }
  # Center seasonal component
  seasonal <- seasonal - mean(seasonal, na.rm = TRUE)

  # Residual
  residual <- x - trend - seasonal

  list(
    observed = x,
    trend = trend,
    seasonal = seasonal,
    residual = residual,
    period = period
  )
}

# Demo: simulate monthly data with trend + seasonality
set.seed(42)
n <- 120  # 10 years of monthly data
t <- 1:n
trend <- 50 + 0.3 * t                        # linear trend
seasonal <- 10 * sin(2 * pi * t / 12)        # yearly cycle
noise <- rnorm(n, 0, 3)
y <- trend + seasonal + noise

result <- decompose_ts(y, period = 12)

cat("=== Time Series Decomposition ===
")
cat(sprintf("  Observations: %d (%.0f years)
", n, n / 12))
cat(sprintf("  Period: %d months
", result$period))

# Show a few values
cat("
  Month  Observed  Trend    Seasonal  Residual
")
for (i in 13:24) {  # year 2
  cat(sprintf("  %3d    %6.1f   %6.1f    %6.1f    %6.1f
",
              i, result$observed[i],
              ifelse(is.na(result$trend[i]), NA, result$trend[i]),
              result$seasonal[i],
              ifelse(is.na(result$residual[i]), NA, result$residual[i])))
}

cat(sprintf("
  Residual SD: %.2f (noise was SD=3)
",
            sd(result$residual, na.rm = TRUE)))
