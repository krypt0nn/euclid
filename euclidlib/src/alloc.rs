#![allow(clippy::result_unit_err, clippy::missing_safety_doc)]

use std::alloc::Layout;

pub trait IntoHeapArray<T, const N: usize> {
    /// Convert current value into a fixed size heap array.
    unsafe fn into_heap_array(self) -> Box<[T; N]>;
}

impl<T, const N: usize> IntoHeapArray<T, N> for [T; N] {
    #[inline]
    unsafe fn into_heap_array(self) -> Box<[T; N]> {
        Box::new(self)
    }
}

impl<'a, T: 'a + Copy, const N: usize> IntoHeapArray<T, N> for &'a [T; N] {
    #[inline]
    unsafe fn into_heap_array(self) -> Box<[T; N]> {
        Box::new(*self)
    }
}

impl<T, const N: usize> IntoHeapArray<T, N> for Box<[T; N]> {
    #[inline(always)]
    unsafe fn into_heap_array(self) -> Box<[T; N]> {
        self
    }
}

impl<T: Clone, const N: usize> IntoHeapArray<T, N> for &Box<[T; N]> {
    #[inline]
    unsafe fn into_heap_array(self) -> Box<[T; N]> {
        self.clone()
    }
}

#[inline]
/// Try to allocate `Box<[T; N]>` directly on heap.
///
/// This function will request zero-allocated memory segment
/// so actually stored `T` values are unsafe to use.
///
/// ```
/// // Allocate `Box<[u8; 4]>` with zero bytes.
/// let array = unsafe {
///     euclidlib::alloc::alloc_fixed_heap_array::<u8, 4>().unwrap()
/// };
///
/// // Build u32 from 4 zero bytes.
/// let num = u32::from_be_bytes(*array);
///
/// // Check that the built u32 is actually 0.
/// assert_eq!(num, 0_u32);
/// ```
pub unsafe fn alloc_fixed_heap_array<T, const N: usize>() -> Result<Box<[T; N]>, ()> {
    let ptr = std::alloc::alloc_zeroed(Layout::new::<[T; N]>());

    if ptr.is_null() {
        return Err(());
    }

    Ok(Box::<[T; N]>::from_raw(ptr as *mut [T; N]))
}

#[inline]
/// Try to allocate `Box<[T; N]>` directly on heap
/// with provided default value.
///
/// ```
/// // Allocate `Box<[u8; 4]>` with `u8::MAX` values (`0b11111111`).
/// let array = unsafe {
///     euclidlib::alloc::alloc_fixed_heap_array_with::<u8, 4>(u8::MAX).unwrap()
/// };
///
/// // Build u32 from 4 `0b11111111` bytes.
/// let num = u32::from_be_bytes(*array);
///
/// // Check that the built u32 is actually `u32::MAX`.
/// assert_eq!(num, u32::MAX);
/// ```
pub unsafe fn alloc_fixed_heap_array_with<T: Copy, const N: usize>(value: T) -> Result<Box<[T; N]>, ()> {
    let mut array = alloc_fixed_heap_array()?;

    for i in 0..N {
        array[i] = value;
    }

    Ok(array)
}

#[inline]
/// Try to allocate `Box<[T; N]>` directly on heap
/// with provided default value.
///
/// ```
/// // Allocate `Box<[u8; 4]>` with `[0, 1, 4, 9]` values.
/// let array = unsafe {
///     euclidlib::alloc::alloc_fixed_heap_array_from::<u8, 4>(|i| i * i).unwrap()
/// };
///
/// // Check that the built box is correct.
/// assert_eq!(&array, &[0, 1, 4, 9]);
/// ```
pub unsafe fn alloc_fixed_heap_array_from<T, const N: usize>(mut callback: impl FnMut(usize) -> T) -> Result<Box<[T; N]>, ()> {
    let mut array = alloc_fixed_heap_array()?;

    for i in 0..N {
        array[i] = callback(i);
    }

    Ok(array)
}
