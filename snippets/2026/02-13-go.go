package main

import (
	"cmp"
	"fmt"
)

func BinarySearch[T cmp.Ordered](arr []T, target T) int {
	lo, hi := 0, len(arr)-1
	for lo <= hi {
		mid := lo + (hi-lo)/2
		switch {
		case arr[mid] == target:
			return mid
		case arr[mid] < target:
			lo = mid + 1
		default:
			hi = mid - 1
		}
	}
	return -1
}

func main() {
	ints := []int{2, 5, 8, 12, 16, 23, 38, 56, 72, 91}
	fmt.Println("Index of 23:", BinarySearch(ints, 23))
	fmt.Println("Index of 42:", BinarySearch(ints, 42))
}
