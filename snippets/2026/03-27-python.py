Matrix = list[list[float]]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two matrices (naive O(n^3) implementation)."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    assert cols_a == rows_b, "incompatible dimensions"
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = mat_mul(A, B)
    for row in C:
        print(row)  # [19, 22] and [43, 50]
