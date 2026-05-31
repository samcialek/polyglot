package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

func worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {
	defer wg.Done()
	for j := range jobs {
		time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)
		fmt.Printf("Worker %d processed job %d\n", id, j)
		results <- j * j
	}
}

func main() {
	const numJobs = 10
	const numWorkers = 3

	jobs := make(chan int, numJobs)
	results := make(chan int, numJobs)

	var wg sync.WaitGroup
	for w := 1; w <= numWorkers; w++ {
		wg.Add(1)
		go worker(w, jobs, results, &wg)
	}

	for j := 1; j <= numJobs; j++ {
		jobs <- j
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	sum := 0
	for r := range results {
		sum += r
	}
	fmt.Printf("Sum of squares: %d\n", sum)
}
