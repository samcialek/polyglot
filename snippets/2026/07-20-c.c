#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int *sieve(int n, int *count) {
    char *is_prime = malloc(n + 1);
    memset(is_prime, 1, n + 1);
    is_prime[0] = is_prime[1] = 0;
    for (int i = 2; i <= (int)sqrt(n); i++)
        if (is_prime[i])
            for (int j = i * i; j <= n; j += i)
                is_prime[j] = 0;
    *count = 0;
    for (int i = 2; i <= n; i++) *count += is_prime[i];
    int *primes = malloc(*count * sizeof(int));
    int idx = 0;
    for (int i = 2; i <= n; i++)
        if (is_prime[i]) primes[idx++] = i;
    free(is_prime);
    return primes;
}

int main(void) {
    int count;
    int *primes = sieve(100, &count);
    printf("Primes up to 100 (%d total):\n", count);
    for (int i = 0; i < count; i++) printf("%d ", primes[i]);
    printf("\n");
    free(primes);
    return 0;
}
