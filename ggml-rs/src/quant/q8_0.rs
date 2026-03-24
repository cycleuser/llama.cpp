//! Q8_0 quantization block

use super::{QuantizationType, Quantize, Dequantize};
use crate::error::Result;
use half::f16;

pub const QK8_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; QK8_0],
}

impl BlockQ8_0 {
    pub fn new() -> Self {
        BlockQ8_0 {
            d: f16::from_f32(0.0),
            qs: [0i8; QK8_0],
        }
    }

    pub fn from_floats(data: &[f32; QK8_0]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in data {
            max_abs = max_abs.max(v.abs());
        }

        let d = max_abs / 127.0;
        let d_f16 = if d == 0.0 { f16::from_f32(0.0) } else { f16::from_f32(d) };

        let mut qs = [0i8; QK8_0];
        if d != 0.0 {
            let id = 1.0 / d;
            for (i, q) in qs.iter_mut().enumerate() {
                let v = (data[i] * id).round().clamp(-127.0, 127.0);
                *q = v as i8;
            }
        }

        BlockQ8_0 { d: d_f16, qs }
    }

    pub fn to_floats(&self) -> [f32; QK8_0] {
        let d = self.d.to_f32();
        let mut out = [0.0f32; QK8_0];

        for (i, &q) in self.qs.iter().enumerate() {
            out[i] = q as f32 * d;
        }

        out
    }
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizationType for BlockQ8_0 {
    #[inline] fn block_size() -> usize { QK8_0 }
    #[inline] fn block_bytes() -> usize { std::mem::size_of::<BlockQ8_0>() }
}

impl Quantize for BlockQ8_0 {
    fn quantize(data: &[f32]) -> Result<Vec<u8>> {
        let nblocks = (data.len() + QK8_0 - 1) / QK8_0;
        let mut blocks = vec![BlockQ8_0::new(); nblocks];

        for (i, block) in blocks.iter_mut().enumerate() {
            let start = i * QK8_0;
            let end = (start + QK8_0).min(data.len());
            let mut block_data = [0.0f32; QK8_0];
            block_data[..end - start].copy_from_slice(&data[start..end]);
            *block = BlockQ8_0::from_floats(&block_data);
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                nblocks * std::mem::size_of::<BlockQ8_0>(),
            )
        };

        Ok(bytes.to_vec())
    }
}

impl Dequantize for BlockQ8_0 {
    fn dequantize(bytes: &[u8], nblocks: usize) -> Result<Vec<f32>> {
        let blocks = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const BlockQ8_0,
                nblocks,
            )
        };

        let mut out = Vec::with_capacity(nblocks * QK8_0);
        for block in blocks {
            let floats = block.to_floats();
            out.extend_from_slice(&floats);
        }

        Ok(out)
    }
}