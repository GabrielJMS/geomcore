//! The single intersection entry point.
//!
//! [`Intersect`] is implemented once per exact pair, in both argument
//! orders, with the pair's own result type as its output. The general
//! results of the untyped path hold their pieces in an [`InlineList`], so
//! no intersection result allocates.

use crate::Tolerance;
use std::ops::{Deref, DerefMut};

/// Exact intersection of two shapes.
///
/// Each implementation returns the result type of its pair, so the
/// compiler knows which outcomes a pair can produce, and dispatch is
/// static. A configuration with no closed form reports "not analytic":
/// no implementation falls back to a numeric intersection on its own.
pub trait Intersect<Rhs: ?Sized = Self> {
    /// The result of intersecting `Self` with `Rhs`.
    type Output;

    /// Intersects `self` with `other`, deciding tangency and coincidence
    /// with `tol`.
    fn intersect(&self, other: &Rhs, tol: Tolerance) -> Self::Output;
}

/// Up to `N` items stored inline, read as a slice.
///
/// Holds the pieces of an intersection result without allocating.
/// Its capacity is a proven bound on the pair it serves, so it never
/// fills up.
#[derive(Clone, Copy)]
pub struct InlineList<T: Copy, const N: usize>(Slots<T, N>);

/// The storage of an [`InlineList`]. The first item pushed fills every
/// slot, so the unused slots always hold a valid value.
#[derive(Clone, Copy)]
enum Slots<T: Copy, const N: usize> {
    Empty,
    Filled { items: [T; N], len: u8 },
}

impl<T: Copy, const N: usize> InlineList<T, N> {
    /// An empty list.
    pub const fn new() -> InlineList<T, N> {
        InlineList(Slots::Empty)
    }

    /// Appends `item`.
    ///
    /// # Panics
    ///
    /// Panics if the list already holds `N` items: the capacity is the
    /// most pieces an intersection can produce, so this is a bug.
    #[cfg_attr(not(test), expect(dead_code, reason = "filled by the general results"))]
    pub(crate) fn push(&mut self, item: T) {
        match &mut self.0 {
            Slots::Empty => {
                assert!(N > 0, "InlineList has no capacity");
                self.0 = Slots::Filled {
                    items: [item; N],
                    len: 1,
                };
            }
            Slots::Filled { items, len } => {
                let at = *len as usize;
                assert!(at < N, "InlineList capacity {N} exceeded");
                items[at] = item;
                *len += 1;
            }
        }
    }
}

impl<T: Copy, const N: usize> Default for InlineList<T, N> {
    fn default() -> InlineList<T, N> {
        InlineList::new()
    }
}

impl<T: Copy, const N: usize> Deref for InlineList<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match &self.0 {
            Slots::Empty => &[],
            Slots::Filled { items, len } => &items[..*len as usize],
        }
    }
}

impl<T: Copy, const N: usize> DerefMut for InlineList<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.0 {
            Slots::Empty => &mut [],
            Slots::Filled { items, len } => &mut items[..*len as usize],
        }
    }
}

impl<T: Copy + PartialEq, const N: usize> PartialEq for InlineList<T, N> {
    fn eq(&self, other: &InlineList<T, N>) -> bool {
        **self == **other
    }
}

impl<T: Copy + std::fmt::Debug, const N: usize> std::fmt::Debug for InlineList<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T: Copy, const N: usize> IntoIterator for &'a InlineList<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_list_empty() {
        let list = InlineList::<f64, 4>::new();
        assert!(list.is_empty());
        assert_eq!(&*list, &[] as &[f64]);
    }

    #[test]
    fn test_inline_list_push_reads_as_slice() {
        let mut list = InlineList::<f64, 4>::new();
        list.push(3.0);
        list.push(1.0);
        list.push(2.0);
        assert_eq!(&*list, &[3.0, 1.0, 2.0]);
        list.sort_by(f64::total_cmp);
        assert_eq!(&*list, &[1.0, 2.0, 3.0]);
        assert_eq!((&list).into_iter().count(), 3);
    }

    #[test]
    fn test_inline_list_fills_to_capacity() {
        let mut list = InlineList::<u8, 4>::new();
        for i in 0..4 {
            list.push(i);
        }
        assert_eq!(&*list, &[0, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "capacity 4 exceeded")]
    fn test_inline_list_overflow_panics() {
        let mut list = InlineList::<u8, 4>::new();
        for i in 0..5 {
            list.push(i);
        }
    }

    #[test]
    fn test_inline_list_equality_ignores_spare_slots() {
        // The spare slots hold copies of the first item pushed, which
        // differ between these two lists.
        let mut a = InlineList::<u8, 4>::new();
        a.push(1);
        a.push(2);
        let mut b = InlineList::<u8, 4>::new();
        b.push(2);
        b.push(1);
        b.sort();
        assert_eq!(a, b);
        assert_ne!(a, InlineList::new());
        assert_eq!(format!("{a:?}"), "[1, 2]");
    }

    #[test]
    fn test_inline_list_adds_no_more_than_a_word() {
        // The empty case costs at most one tag word over the items.
        assert!(std::mem::size_of::<InlineList<[f64; 3], 4>>() <= 4 * 24 + 8);
    }
}
