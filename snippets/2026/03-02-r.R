# Custom statistical summary functions — beyond base R.

geometric_mean <- function(x) {
  exp(mean(log(x[x > 0])))
}

harmonic_mean <- function(x) {
  n <- length(x[x != 0])
  n / sum(1 / x[x != 0])
}

trimmed_summary <- function(x, trim = 0.1) {
  sorted <- sort(x)
  n <- length(sorted)
  lo <- floor(n * trim) + 1
  hi <- n - floor(n * trim)
  trimmed <- sorted[lo:hi]

  list(
    n = length(x),
    trimmed_n = length(trimmed),
    mean = mean(trimmed),
    sd = sd(trimmed),
    median = median(trimmed),
    iqr = IQR(trimmed),
    geometric_mean = geometric_mean(trimmed),
    harmonic_mean = harmonic_mean(trimmed)
  )
}

# Demo with simulated data
set.seed(42)
data <- c(rnorm(95, mean = 50, sd = 10), runif(5, 200, 500))  # 5 outliers

cat("=== Raw data summary ===
")
print(summary(data))

cat("
=== Trimmed (10%) summary ===
")
result <- trimmed_summary(data, trim = 0.1)
for (name in names(result)) {
  cat(sprintf("  %-20s %s
", paste0(name, ":"), round(result[[name]], 3)))
}
