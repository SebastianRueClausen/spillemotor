use std::mem::ManuallyDrop;
use std::ops::Deref;

struct RefCounts {
    counts: Vec<u16>,
    free_slots: Vec<u32>,
}

impl RefCounts {
    const fn new() -> Self {
        Self { counts: Vec::new(), free_slots: Vec::new() }
    }
}

impl RefCounts {
    fn new_ref_count(&mut self) -> u32 {
        if let Some(slot) = self.free_slots.pop() {
            self.counts[slot as usize] = 1;
            slot
        } else {
            self.counts.push(1);
            self.counts.len() as u32 - 1
        }
    }

    fn increase_count(&mut self, slot: u32) {
        self.counts[slot as usize] += 1;
    }

    fn decrease_count(&mut self, slot: u32) -> bool {
        let count = &mut self.counts[slot as usize];

        *count -= 1;

        if *count == 0 {
            self.free_slots.push(slot);
            true
        } else {
            false
        }
    }
}

static mut REF_COUNTS: RefCounts = RefCounts::new();

pub struct Handle<T: Clone> {
    val: ManuallyDrop<T>, 
    ref_count: u32,
}

impl<T: Clone> Handle<T> {
    pub fn new(val: T) -> Self {
        let ref_count = unsafe { REF_COUNTS.new_ref_count() };
        let val = ManuallyDrop::new(val);
        Self { val, ref_count }
    }
}

impl<T: Clone> Clone for Handle<T> {
    fn clone(&self) -> Self {
        unsafe { REF_COUNTS.increase_count(self.ref_count); }
        Self { ref_count: self.ref_count, val: self.val.clone() }
    }
}

impl<T: Clone> Deref for Handle<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.val 
    }
}

impl<T: Clone> Drop for Handle<T> {
    fn drop(&mut self) {
        unsafe {
            if REF_COUNTS.decrease_count(self.ref_count) {
                ManuallyDrop::drop(&mut self.val);
            }
        }
    }
}

#[test]
fn simple() {
    let s1 = Handle::new(1);
    let s2 = s1.clone();
    let s3 = s2.clone();

    assert_eq!(*s2, *s3); 
}

#[test]
fn drop() {
    #[derive(Clone)]
    struct Test;
    static mut COUNT: u32 = 0;
    impl Drop for Test {
        fn drop(&mut self) {
            unsafe { COUNT += 1; }
        }
    }
    {
        let s1 = Handle::new(Test);
        {
            let s2 = s1.clone();
            let s2 = s1.clone();
            let s3 = s1.clone();
        }
        assert_eq!(unsafe { COUNT }, 0);
    }
    assert_eq!(unsafe { COUNT }, 1);
}
