package main

import "fmt"

type RingBuffer[T any] struct {
	buf   []T
	size  int
	head  int
	tail  int
	count int
}

func NewRingBuffer[T any](capacity int) *RingBuffer[T] {
	return &RingBuffer[T]{buf: make([]T, capacity), size: capacity}
}

func (r *RingBuffer[T]) Push(v T) bool {
	if r.count == r.size { return false }
	r.buf[r.tail] = v
	r.tail = (r.tail + 1) % r.size
	r.count++
	return true
}

func (r *RingBuffer[T]) Pop() (T, bool) {
	var zero T
	if r.count == 0 { return zero, false }
	v := r.buf[r.head]
	r.head = (r.head + 1) % r.size
	r.count--
	return v, true
}

func main() {
	rb := NewRingBuffer[int](3)
	rb.Push(10); rb.Push(20); rb.Push(30)
	v, _ := rb.Pop()
	fmt.Println("Popped:", v)
	rb.Push(40)
	for rb.count > 0 {
		v, _ := rb.Pop()
		fmt.Println(v)
	}
}
