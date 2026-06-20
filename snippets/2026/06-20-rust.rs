fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let (mut lo, mut hi) = (0, arr.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match arr[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    None
}

fn main() {
    let data = vec![2, 5, 8, 12, 16, 23, 38, 56, 72, 91];
    println!("Search 23: {:?}", binary_search(&data, &23));
    println!("Search 42: {:?}", binary_search(&data, &42));
}
