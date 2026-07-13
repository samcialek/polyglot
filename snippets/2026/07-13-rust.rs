use std::collections::HashMap;

fn word_freq(text: &str) -> HashMap<String, usize> {
    let mut freq = HashMap::new();
    for word in text.split_whitespace() {
        let clean: String = word.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        if !clean.is_empty() {
            *freq.entry(clean).or_insert(0) += 1;
        }
    }
    freq
}

fn main() {
    let text = "the quick brown fox jumps over the lazy dog the fox";
    let freq = word_freq(text);
    let mut pairs: Vec<_> = freq.iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(a.1));
    for (word, count) in &pairs[..5.min(pairs.len())] {
        println!("{:>8}: {}", word, count);
    }
}
