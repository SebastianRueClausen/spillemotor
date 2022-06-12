
// From: https://users.rust-lang.org/t/can-i-conveniently-compile-bytes-into-a-rust-program-with-a-specific-alignment/24049/2.

#[repr(C)]
#[allow(dead_code)]
pub struct AlignedAs<Align, Bytes: ?Sized> {
    pub _align: [Align; 0],
    pub bytes: Bytes,
}

macro_rules! include_bytes_aligned_as {
    ($align_ty:ty, $path:literal) => {
        {
            use $crate::macros::AlignedAs;
           
            #[allow(dead_code)]
            static ALIGNED: &AlignedAs::<$align_ty, [u8]> = &AlignedAs {
                _align: [],
                bytes: *include_bytes!($path),
            };

            &ALIGNED.bytes
        }
    };
}

