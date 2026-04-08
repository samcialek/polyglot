struct Fibonacci { a: u64, b: u64 }

impl Fibonacci {
    fn new() -> Self { Fibonacci { a: 0, b: 1 } }
}

impl Iterator for Fibonacci {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        let val = self.a;
        (self.a, self.b) = (self.b, self.a + self.b);
        Some(val)
    }
}

fn main() {
    let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
    println!("Fibonacci: {:?}", fibs);

    let sum: u64 = Fibonacci::new()
        .take_while(|&n| n < 4_000_000)
        .filter(|n| n % 2 == 0)
        .sum();
    println!("Sum of even fibs < 4M: {}", sum);
}
