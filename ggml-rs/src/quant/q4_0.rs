//! Q4_0 quantization block

use super::{QuantizationType, Quantize, Dequantize};
use crate::error::Result;
use half::f16;

pub const QK4_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; QK4_0 / 2],
}

impl BlockQ4_0 {
    pub fn new() -> Self {
        BlockQ4_0 {
            d: f16::from_f32(0.0),
            qs: [0u8; QK4_0 / 2],
        }
    }

    pub fn from_floats(data: &[f32; QK4_0]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in data {
            max_abs = max_abs.max(v.abs());
        }

        let d = max_abs / -8.0;
        let d_f16 = if d == 0.0 { f16::from_f32(0.0) } else { f16::from_f32(d) };

        let mut qs = [0u8; QK4_0 / 2];
        if d != 0.0 {
            let id = 1.0 / d;
            for i in 0..QK4_0 / 2 {
                let v0 = (data[i * 2] * id + 8.5).clamp(0.0, 15.0) as u8;
                let v1 = (data[i * 2 + 1] * id + 8.5).clamp(0.0, 15.0) as u8;
                qs[i] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
            }
        }

        BlockQ4_0 { d: d_f16, qs }
    }

    pub fn to_floats(&self) -> [f32; QK4_0] {
        let d = self.d.to_f32();
        let mut out = [0.0f32; QK4_0];

        for i in 0..QK4_0 / 2 {
            let v0 = (self.qs[i] & 0x0F) as i32 - 8;
            let v1 = ((self.qs[i] >> 4) & 0x0F) as i32 - 8;
            out[i * 2] = v0 as f32 * d;
            out[i * 2 + 1] = v1 as f32 * d;
        }

        out
    }
}

impl Default for BlockQ4_0 {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizationType for BlockQ4_0 {
    #[inline] fn block_size() -> usize { QK4_0 }
    #[inline] fn block_bytes() -> usize { std::mem::size_of::<BlockQ4_0>() }
}

impl Quantize for BlockQ4_0 {
    fn quantize(data: &[f32]) -> Result<Vec<u8>> {
        let nblocks = (data.len() + QK4_0 - 1) / QK4_0;
        let mut blocks = vec![BlockQ4_0::new(); nblocks];

        for (i, block) in blocks.iter_mut().enumerate() {
            let start = i * QK4_0;
            let end = (start + QK4_0).min(data.len());
            let mut block_data = [0.0f32; QK4_0];
            block_data[..end - start].copy_from_slice(&data[start..end]);
            *block = BlockQ4_0::from_floats(&block_data);
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                nblocks * std::mem::size_of::<BlockQ4_0>(),
            )
        };

        Ok(bytes.to_vec())
    }
}

impl Dequantize for BlockQ4_0 {
    fn dequantize(bytes: &[u8], nblocks: usize) -> Result<Vec<f32>> {
        let blocks = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const BlockQ4_0,
                nblocks,
            )
        };

        let mut out = Vec::with_capacity(nblocks * QK4_0);
        for block in blocks {
            let floats = block.to_floats();
            out.extend_from_slice(&floats);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip() {
        let data: [f32; QK4_0] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
            0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625,
            10.0, -10.0, 5.0, -5.0, 2.5, -2.5, 1.25, -1.25,
        ];

        let block = BlockQ4_0::from_floats(&data);
        let recovered = block.to_floats();

        for (i, (orig, recov)) in data.iter().zip(recovered.iter()).enumerate() {
            let rel_error = (orig - recov).abs() / orig.abs().max(0.001);
            assert!(rel_error < 0.15, "Mismatch at index {}: {} vs {} (rel_error={})", i, orig, recov, rel_error);
        }
    }
}