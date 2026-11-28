/**
 * In-place quicksort using Lomuto partition scheme.
 */
function quickSort(arr, lo = 0, hi = arr.length - 1) {
  if (lo < hi) {
    const pivot = partition(arr, lo, hi);
    quickSort(arr, lo, pivot - 1);
    quickSort(arr, pivot + 1, hi);
  }
  return arr;
}

function partition(arr, lo, hi) {
  const pivot = arr[hi];
  let i = lo;
  for (let j = lo; j < hi; j++) {
    if (arr[j] <= pivot) {
      [arr[i], arr[j]] = [arr[j], arr[i]];
      i++;
    }
  }
  [arr[i], arr[hi]] = [arr[hi], arr[i]];
  return i;
}

console.log(quickSort([10, 7, 8, 9, 1, 5])); // [1, 5, 7, 8, 9, 10]
