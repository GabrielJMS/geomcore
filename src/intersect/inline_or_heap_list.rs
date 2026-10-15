//! A list that keeps up to `N` items inline and moves them to the heap
//! only when it grows past `N`.
//!
//! The routed results store their hits in it: an exact pair
//! never produces more than the capacity, so it never allocates, while a
//! numeric search, whose hit count has no bound, keeps its `Vec`.

use std::ops::{Deref, DerefMut};

use super::InlineList;

/// Up to `N` items inline, any number on the heap, read as a slice.
#[derive(Clone)]
pub(crate) enum InlineOrHeapList<T: Copy, const N: usize> {
    /// At most `N` items, stored in place.
    Inline(InlineList<T, N>),
    /// Any number of items, stored on the heap.
    Heap(Vec<T>),
}

impl<T: Copy, const N: usize> InlineOrHeapList<T, N> {
    /// An empty list; it does not allocate.
    pub(crate) const fn new() -> InlineOrHeapList<T, N> {
        InlineOrHeapList::Inline(InlineList::new())
    }

    /// Appends `item`, moving every item to the heap when the inline
    /// storage is full.
    #[inline]
    pub(crate) fn push(&mut self, item: T) {
        if let InlineOrHeapList::Inline(list) = self {
            if list.len() < N {
                list.push(item);
                return;
            }
            let mut spilled = Vec::with_capacity(2 * N + 1);
            spilled.extend_from_slice(list);
            *self = InlineOrHeapList::Heap(spilled);
        }
        if let InlineOrHeapList::Heap(items) = self {
            items.push(item);
        }
    }
}

impl<T: Copy, const N: usize> From<Vec<T>> for InlineOrHeapList<T, N> {
    /// Keeps the vector's buffer as it is: no copy.
    fn from(items: Vec<T>) -> InlineOrHeapList<T, N> {
        InlineOrHeapList::Heap(items)
    }
}

impl<T: Copy, const N: usize> FromIterator<T> for InlineOrHeapList<T, N> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> InlineOrHeapList<T, N> {
        let mut list = InlineOrHeapList::new();
        for item in items {
            list.push(item);
        }
        list
    }
}

impl<T: Copy, const N: usize> Deref for InlineOrHeapList<T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        match self {
            InlineOrHeapList::Inline(list) => list,
            InlineOrHeapList::Heap(items) => items,
        }
    }
}

impl<T: Copy, const N: usize> DerefMut for InlineOrHeapList<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            InlineOrHeapList::Inline(list) => list,
            InlineOrHeapList::Heap(items) => items,
        }
    }
}

/// Equal when the items are equal, wherever they are stored.
impl<T: Copy + PartialEq, const N: usize> PartialEq for InlineOrHeapList<T, N> {
    fn eq(&self, other: &InlineOrHeapList<T, N>) -> bool {
        **self == **other
    }
}

/// Prints the items as a list, like a `Vec`.
impl<T: Copy + std::fmt::Debug, const N: usize> std::fmt::Debug for InlineOrHeapList<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stays_inline_up_to_capacity() {
        let list: InlineOrHeapList<u32, 3> = [1, 2, 3].into_iter().collect();
        assert!(matches!(list, InlineOrHeapList::Inline(_)));
        assert_eq!(*list, [1, 2, 3]);
    }

    #[test]
    fn test_moves_to_the_heap_past_capacity_and_keeps_order() {
        let list: InlineOrHeapList<u32, 2> = [1, 2, 3, 4, 5].into_iter().collect();
        assert!(matches!(list, InlineOrHeapList::Heap(_)));
        assert_eq!(*list, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_equality_ignores_the_storage() {
        let inline: InlineOrHeapList<u32, 4> = [3, 1].into_iter().collect();
        let heap: InlineOrHeapList<u32, 4> = vec![3, 1].into();
        assert_eq!(inline, heap);
        assert_eq!(format!("{inline:?}"), format!("{:?}", vec![3, 1]));
    }

    #[test]
    fn test_sorts_in_place_through_the_slice() {
        let mut list: InlineOrHeapList<u32, 4> = [3, 1, 2].into_iter().collect();
        list.sort_unstable();
        assert_eq!(*list, [1, 2, 3]);
        assert!(InlineOrHeapList::<u32, 4>::new().is_empty());
    }
}
