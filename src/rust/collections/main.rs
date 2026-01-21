use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

fn main() {
    // Vec
    let mut vec = vec![1, 2, 3];
    vec.push(4);
    println!("Vec: {:?}", vec);

    // VecDeque (like std::deque)
    let mut deque = VecDeque::new();
    deque.push_back(1);
    deque.push_front(0);
    println!("VecDeque: {:?}", deque);

    // HashMap (like std::unordered_map)
    let mut hash = HashMap::new();
    hash.insert("one", 1);
    hash.insert("two", 2);
    println!("HashMap: {:?}", hash);

    // BTreeMap (like std::map)
    let mut ordered = BTreeMap::new();
    ordered.insert("one", 1);
    ordered.insert("two", 2);
    println!("BTreeMap: {:?}", ordered);

    // HashSet (like std::unordered_set)
    let mut set = HashSet::new();
    set.insert(1);
    set.insert(2);
    set.insert(1); // duplicate ignored
    println!("HashSet: {:?}", set);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec() {
        let mut v = vec![1, 2, 3];
        v.push(4);
        assert_eq!(v, vec![1, 2, 3, 4]);
        assert_eq!(v.pop(), Some(4));
    }

    #[test]
    fn test_vecdeque() {
        let mut d = VecDeque::new();
        d.push_back(1);
        d.push_front(0);
        assert_eq!(d.pop_front(), Some(0));
        assert_eq!(d.pop_back(), Some(1));
    }

    #[test]
    fn test_hashmap() {
        let mut m = HashMap::new();
        m.insert("key", 42);
        assert_eq!(m.get("key"), Some(&42));
        assert_eq!(m.get("missing"), None);
    }

    #[test]
    fn test_btreemap() {
        let mut m = BTreeMap::new();
        m.insert(2, "two");
        m.insert(1, "one");
        let keys: Vec<_> = m.keys().collect();
        assert_eq!(keys, vec![&1, &2]); // sorted
    }

    #[test]
    fn test_hashset() {
        let mut s = HashSet::new();
        assert!(s.insert(1));
        assert!(!s.insert(1)); // duplicate
        assert!(s.contains(&1));
    }
}
