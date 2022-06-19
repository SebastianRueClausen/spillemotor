
/// Kernighans algorithm.
#[inline]
fn is_pow2(a: u64) -> bool {
    a != 0 && (a & (a - 1)) == 0
}

/// Calculate least common multiple for two powers of 2.
#[inline]
pub fn lcm(a: u64, b: u64) -> u64 {
    if is_pow2(a) && is_pow2(b) {
        a.max(b)
    } else {
        a * (b / gcd(a, b))
    }
}

/// Calculate greatest common divisor.
#[inline]
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    // Optimize if `a` and `b` are both powers of 2.
    if is_pow2(a) && is_pow2(b) {
        a.min(b)
    } else {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }
}

/// Round `a` up to next integer with aligned to `aligment`.
#[inline]
pub fn align_up_to(a: u64, alignment: u64) -> u64 {
    ((a + alignment - 1) / alignment) * alignment
}

#[test]
fn test_lcm() {
    assert_eq!(lcm(5, 12), 60);
    assert_eq!(lcm(4, 64), 64);
    assert_eq!(lcm(2, 5), 10);
    assert_eq!(lcm(3, 4), 12);
    assert_eq!(lcm(2, 4), 4);
}

#[test]
fn test_gcd() {
    assert_eq!(gcd(5, 12), 1);
    assert_eq!(gcd(4, 64), 4);
    assert_eq!(gcd(2, 5), 1);
    assert_eq!(gcd(3, 4), 1);
    assert_eq!(gcd(2, 4), 2);
    assert_eq!(gcd(10, 12), 2);
}
