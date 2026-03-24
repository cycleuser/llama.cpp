//! Q4_1 quantization block

use super::{QuantizationType, Quantize, Dequantize};
use crate::error::Result;
use half::f16;

pub const QK4_1: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_1 {
    pub d: f16,
    pub m: f16,
    pub qs: [u8; QK4_1 / 2],
}

impl BlockQ4_1 {
    pub fn new() -> Self {
        BlockQ4_1 {
            d: f16::from_f32(0.0),
            m: f16::from_f32(0.0),
            qs: [0u8; QK4_1 / 2],
        }
    }

    pub fn from_floats(data: &[f32; QK4_1]) -> Self {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &v in data {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        let d = (max_val - min_val) / 15.0;
        let m = min_val;

        let d_f16 = if d == 0.0 { f16::from_f32(0.0) } else { f16::from_f32(d) };
        let m_f16 = f16::from_f32(m);

        let mut qs = [0u8; QK4_1 / 2];
        if d != 0.0 {
            let id = 1.0 / d;
            for i in 0..QK4_1 / 2 {
                let v0 = ((data[i * 2] - m) * id + 0.5).clamp(0.0, 15.0) as u8;
                let v1 = ((data[i * 2 + 1] - m) * id + 0.5).clamp(0.0, 15.0) as u8;
                qs[i] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
            }
        }

        BlockQ4_1 { d: d_f16, m: m_f16, qs }
    }

    pub fn to_floats(&self) -> [f32; QK4_1] {
        let d = self.d.to_f32();
        let m = self.m.to_f32();
        let mut out = [0.0f32; QK4_1];

        for i in 0..QK4_1 / 2 {
            let v0 = (self.qs[i] & 0x0F) as f32;
            let v1 = ((self.qs[i] >> 4) & 0x0F) as f32;
            out[i * 2] = v0 * d + m;
            out[i * 2 + 1] = v1 * d + m;
        }

        out
    }
}

impl Default for BlockQ4_1 {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizationType for BlockQ4_1 {
    #[inline] fn block_size() -> usize { QK4_1 }
    #[inline] fn block_bytes() -> usize { std::mem::size_of::<BlockQ4_1>() }
}

impl Quantize for BlockQ4_1 {
    fn quantize(data: &[f32]) -> Result<Vec<u8>> {
        let nblocks = (data.len() + QK4_1 - 1) / QK4_1;
        let mut blocks = vec![BlockQ4_1::new(); nblocks];

        for (i, block) in blocks.iter_mut().enumerate() {
            let start = i * QK4_1;
            let end = (start + QK4_1).min(data.len());
            let mut block_data = [0.0f32; QK4_1];
            block_data[..end - start].copy_from_slice(&data[start..end]);
            *block = BlockQ4_1::from_floats(&block_data);
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                nblocks * std::mem::size_of::<BlockQ4_1>(),
            )
        };

        Ok(bytes.to_vec())
    }
}

impl Dequantize for BlockQ4_1 {
    fn dequantize(bytes: &[u8], nblocks: usize) -> Result<Vec<f32>> {
        let blocks = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const BlockQ4_1,
                nblocks,
            )
        };

        let mut out = Vec::with_capacity(nblocks * QK4_1);
        for block in blocks {
            let floats = block.to_floats();
            out.extend_from_slice(&floats);
        }

        Ok(out)
    }
}