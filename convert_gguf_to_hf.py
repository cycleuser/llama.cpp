#!/usr/bin/env python3
"""
Standalone GGUF to HuggingFace converter.

No dependencies on llama.cpp or gguf-py. Only requires:
  - numpy
  - torch or safetensors (for output)

Usage:
    python convert_gguf_to_hf.py model.gguf -o hf_model/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from collections import OrderedDict
from enum import IntEnum
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

logger = logging.getLogger("gguf-to-hf")

# ============================================================
# Constants (from GGUF specification)
# ============================================================

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

QK_K = 256


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 38
    NVFP4 = 39


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG = 1


# Block size and type size for each quantization type
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32:     (1, 4),
    GGMLQuantizationType.F16:     (1, 2),
    GGMLQuantizationType.Q4_0:    (32, 18),
    GGMLQuantizationType.Q4_1:    (32, 20),
    GGMLQuantizationType.Q5_0:    (32, 22),
    GGMLQuantizationType.Q5_1:    (32, 24),
    GGMLQuantizationType.Q8_0:    (32, 34),
    GGMLQuantizationType.Q8_1:    (32, 36),
    GGMLQuantizationType.Q2_K:    (256, 84),
    GGMLQuantizationType.Q3_K:    (256, 110),
    GGMLQuantizationType.Q4_K:    (256, 144),
    GGMLQuantizationType.Q5_K:    (256, 176),
    GGMLQuantizationType.Q6_K:    (256, 210),
    GGMLQuantizationType.IQ2_XXS: (256, 66),
    GGMLQuantizationType.IQ2_XS:  (256, 74),
    GGMLQuantizationType.IQ3_XXS: (256, 66),
    GGMLQuantizationType.IQ1_S:   (256, 50),
    GGMLQuantizationType.IQ4_NL:  (32, 18),
    GGMLQuantizationType.IQ3_S:   (256, 110),
    GGMLQuantizationType.IQ2_S:   (256, 82),
    GGMLQuantizationType.IQ4_XS:  (256, 144),
    GGMLQuantizationType.I8:      (1, 1),
    GGMLQuantizationType.I16:     (1, 2),
    GGMLQuantizationType.I32:     (1, 4),
    GGMLQuantizationType.I64:     (1, 8),
    GGMLQuantizationType.F64:     (1, 8),
    GGMLQuantizationType.IQ1_M:   (256, 54),
    GGMLQuantizationType.BF16:    (1, 2),
    GGMLQuantizationType.TQ1_0:   (256, 110),
    GGMLQuantizationType.TQ2_0:   (256, 82),
    GGMLQuantizationType.MXFP4:   (32, 9),
    GGMLQuantizationType.NVFP4:   (64, 36),
}


# ============================================================
# Dequantization (simplified - only common types needed for training)
# ============================================================

def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    return (*shape[:-1], shape[-1] // block_size * type_size)


def dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    """Dequantize a tensor to FP32."""
    if qtype == GGMLQuantizationType.F32:
        return data.view(np.float32)
    elif qtype == GGMLQuantizationType.F16:
        return data.view(np.float16).astype(np.float32)
    elif qtype == GGMLQuantizationType.BF16:
        return (data.view(np.int16).astype(np.int32) << 16).view(np.float32)
    elif qtype == GGMLQuantizationType.F64:
        return data.view(np.float64)
    elif qtype == GGMLQuantizationType.Q4_0:
        return _dequant_q4_0(data)
    elif qtype == GGMLQuantizationType.Q4_1:
        return _dequant_q4_1(data)
    elif qtype == GGMLQuantizationType.Q5_0:
        return _dequant_q5_0(data)
    elif qtype == GGMLQuantizationType.Q5_1:
        return _dequant_q5_1(data)
    elif qtype == GGMLQuantizationType.Q8_0:
        return _dequant_q8_0(data)
    elif qtype == GGMLQuantizationType.Q2_K:
        return _dequant_q2_k(data)
    elif qtype == GGMLQuantizationType.Q3_K:
        return _dequant_q3_k(data)
    elif qtype == GGMLQuantizationType.Q4_K:
        return _dequant_q4_k(data)
    elif qtype == GGMLQuantizationType.Q5_K:
        return _dequant_q5_k(data)
    elif qtype == GGMLQuantizationType.Q6_K:
        return _dequant_q6_k(data)
    elif qtype == GGMLQuantizationType.IQ4_NL:
        return _dequant_iq4_nl(data)
    elif qtype == GGMLQuantizationType.MXFP4:
        return _dequant_mxfp4(data)
    elif qtype == GGMLQuantizationType.NVFP4:
        return _dequant_nvfp4(data)
    else:
        raise NotImplementedError(f"Dequantization for {qtype.name} is not yet implemented")


def _dequant_q4_0(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 18
    blocks = data.reshape((n_blocks, 18))
    d, qs = np.hsplit(blocks, [2])
    d = d.view(np.float16).astype(np.float32)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)
    return (d * qs.astype(np.float32)).reshape(-1)


def _dequant_q4_1(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 20
    blocks = data.reshape((n_blocks, 20))
    d, rest = np.hsplit(blocks, [2])
    m, qs = np.hsplit(rest, [2])
    d = d.view(np.float16).astype(np.float32)
    m = m.view(np.float16).astype(np.float32)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.float32)
    return ((d * qs) + m).reshape(-1)


def _dequant_q5_0(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 22
    blocks = data.reshape((n_blocks, 22))
    d, rest = np.hsplit(blocks, [2])
    qh, qs = np.hsplit(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    qh = qh.view(np.uint32)
    qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
    ql = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = (qh & np.uint32(0x01)).astype(np.uint8)
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))
    qs = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)
    return (d * qs.astype(np.float32)).reshape(-1)


def _dequant_q5_1(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 24
    blocks = data.reshape((n_blocks, 24))
    d, rest = np.hsplit(blocks, [2])
    m, rest = np.hsplit(rest, [2])
    qh, qs = np.hsplit(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    m = m.view(np.float16).astype(np.float32)
    qh = qh.view(np.uint32)
    qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
    ql = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = (qh & np.uint32(0x01)).astype(np.uint8)
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))
    qs = (ql | (qh << np.uint8(4))).astype(np.float32)
    return ((d * qs) + m).reshape(-1)


def _dequant_q8_0(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 34
    blocks = data.reshape((n_blocks, 34))
    d, x = np.split(blocks, [2], axis=1)
    d = d.view(np.float16).astype(np.float32)
    x = x.view(np.int8).astype(np.float32)
    return (x * d).reshape(-1)


def _dequant_q2_k(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 84
    blocks = data.reshape((n_blocks, 84))
    scales, rest = np.hsplit(blocks, [16])
    qs, rest = np.hsplit(rest, [64])
    d, dmin = np.hsplit(rest, [2])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    dl = (d * (scales & 0xF).astype(np.float32)).reshape((n_blocks, 16, 1))
    ml = (dmin * (scales >> 4).astype(np.float32)).reshape((n_blocks, 16, 1))
    shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & np.uint8(3)
    qs = qs.reshape((n_blocks, 16, 16)).astype(np.float32)
    return (dl * qs - ml).reshape(-1)


def _dequant_q3_k(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 110
    blocks = data.reshape((n_blocks, 110))
    hmask, rest = np.hsplit(blocks, [32])
    qs, rest = np.hsplit(rest, [64])
    scales, d = np.hsplit(rest, [12])
    d = d.view(np.float16).astype(np.float32)
    lscales, hscales = np.hsplit(scales, [8])
    lscales = lscales.reshape((n_blocks, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & np.uint8(0x0F)) | ((hscales & np.uint8(0x03)) << np.uint8(4))
    scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, 16)) & np.uint8(3)
    qh = (qh.reshape((n_blocks, 16, 16)) & np.uint8(1))
    qh = qh ^ np.uint8(1)
    q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)
    return (dl * q).reshape(-1)


def _get_scale_min_q4k(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = scales.shape[0]
    scales = scales.view(np.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return (sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8)))


def _dequant_q4_k(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 144
    blocks = data.reshape((n_blocks, 144))
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [12])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, m = _get_scale_min_q4k(scales)
    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)
    return (d * qs - dm).reshape(-1)


def _dequant_q5_k(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 176
    blocks = data.reshape((n_blocks, 176))
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, rest = np.hsplit(rest, [12])
    qh, qs = np.hsplit(rest, [32])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, m = _get_scale_min_q4k(scales)
    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh = (qh & np.uint8(0x01)).reshape((n_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.float32)
    return (d * q - dm).reshape(-1)


def _dequant_q6_k(data: np.ndarray) -> np.ndarray:
    n_blocks = data.size // 210
    blocks = data.reshape((n_blocks, 210))
    ql, rest = np.hsplit(blocks, [128])
    qh, rest = np.hsplit(rest, [64])
    scales, d = np.hsplit(rest, [16])
    scales = scales.view(np.int8).astype(np.float32)
    d = d.view(np.float16).astype(np.float32)
    d = (d * scales).reshape((n_blocks, 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape((n_blocks, 16, -1)).astype(np.float32)
    return (d * q).reshape(-1)


def _dequant_iq4_nl(data: np.ndarray) -> np.ndarray:
    """IQ4_NL dequantization (simplified 4-bit lookup)."""
    n_blocks = data.size // 18
    blocks = data.reshape((n_blocks, 18))
    d, qs = np.hsplit(blocks, [2])
    d = d.view(np.float16).astype(np.float32)
    # IQ4_NL uses a simple 4-bit quantization with lookup table
    qs_lo = (qs & 0x0F).astype(np.int8) - 8
    qs_hi = (qs >> 4).astype(np.int8) - 8
    qs = np.concatenate([qs_lo, qs_hi], axis=-1).astype(np.float32)
    return (d * qs).reshape(-1)


def _e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
    bits = np.where(x < 2, np.uint32(0x00200000) << np.uint32(x), np.uint32(x - 1) << np.uint32(23))
    return bits.view(np.float32)


def _dequant_mxfp4(data: np.ndarray) -> np.ndarray:
    kvalues = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
    n_blocks = data.size // 9
    blocks = data.reshape((n_blocks, 9))
    e, qs = np.hsplit(blocks, [1])
    d = _e8m0_to_fp32_half(e)
    qs = qs.reshape((n_blocks, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
    qs = (qs & np.uint8(0x0F)).view(np.int8)
    kvalues = np.array(kvalues, dtype=np.int8).reshape(1, 1, 16)
    qs = np.take_along_axis(kvalues, qs, axis=-1).reshape((n_blocks, 32))
    return (d * qs.astype(np.float32)).reshape(-1)


def _ue4m3_to_fp32(x: np.ndarray) -> np.ndarray:
    exp = (x >> 3).astype(np.int32) & 0xF
    man = (x & 0x7).astype(np.float32)
    raw = np.where(exp == 0, man * 2**-9, (1.0 + man / 8.0) * (2.0 ** (exp.astype(np.float32) - 7)))
    return np.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5)


def _dequant_nvfp4(data: np.ndarray) -> np.ndarray:
    kvalues = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
    n_super = data.size // 36
    blocks = data.reshape((n_super, 36))
    d_bytes, qs = np.hsplit(blocks, [4])
    d = _ue4m3_to_fp32(d_bytes).reshape(n_super, 4, 1)
    qs = qs.reshape(n_super, 4, 8)
    lo = (qs & np.uint8(0x0F)).view(np.int8)
    hi = (qs >> np.uint8(4)).view(np.int8)
    vals = np.concatenate([lo, hi], axis=-1)
    kvalues = np.array(kvalues, dtype=np.int8).reshape(1, 1, 16)
    vals = np.take_along_axis(kvalues, vals, axis=-1)
    return (d * vals.astype(np.float32)).reshape(n_super, 64).reshape(-1)


# ============================================================
# GGUF Reader (standalone)
# ============================================================

class ReaderField(NamedTuple):
    offset: int
    name: str
    parts: list[np.ndarray] = []
    data: list[int] = [-1]
    types: list[GGUFValueType] = []

    def contents(self, index_or_slice: int | slice = slice(None)) -> Any:
        if self.types:
            to_string = lambda x: str(x.tobytes(), encoding='utf-8')
            main_type = self.types[0]
            if main_type == GGUFValueType.ARRAY:
                sub_type = self.types[-1]
                if sub_type == GGUFValueType.STRING:
                    indices = self.data[index_or_slice]
                    if isinstance(index_or_slice, int):
                        return to_string(self.parts[indices])
                    else:
                        return [to_string(self.parts[idx]) for idx in indices]
                else:
                    if isinstance(index_or_slice, int):
                        return self.parts[self.data[index_or_slice]].tolist()[0]
                    else:
                        return [pv for idx in self.data[index_or_slice] for pv in self.parts[idx].tolist()]
            if main_type == GGUFValueType.STRING:
                return to_string(self.parts[-1])
            else:
                return self.parts[-1].tolist()[0]
        return None


class ReaderTensor(NamedTuple):
    name: str
    tensor_type: GGMLQuantizationType
    shape: np.ndarray
    n_elements: int
    n_bytes: int
    data_offset: int
    data: np.ndarray
    field: ReaderField


class GGUFReader:
    byte_order: str = 'I'
    alignment: int = GGUF_DEFAULT_ALIGNMENT
    data_offset: int

    gguf_scalar_to_np: dict[GGUFValueType, type] = {
        GGUFValueType.UINT8:   np.uint8,
        GGUFValueType.INT8:    np.int8,
        GGUFValueType.UINT16:  np.uint16,
        GGUFValueType.INT16:   np.int16,
        GGUFValueType.UINT32:  np.uint32,
        GGUFValueType.INT32:   np.int32,
        GGUFValueType.FLOAT32: np.float32,
        GGUFValueType.BOOL:    np.bool_,
        GGUFValueType.UINT64:  np.uint64,
        GGUFValueType.INT64:   np.int64,
        GGUFValueType.FLOAT64: np.float64,
    }

    def __init__(self, path: str | Path, mode: str = 'r'):
        self.data = np.memmap(path, mode=mode)
        offs = 0

        if self._get(offs, np.uint32, override_order='<')[0] != GGUF_MAGIC:
            raise ValueError('GGUF magic invalid')
        offs += 4

        temp_version = self._get(offs, np.uint32)
        if temp_version[0] & 65535 == 0:
            self.byte_order = 'S'
            temp_version = temp_version.view(temp_version.dtype.newbyteorder(self.byte_order))
        version = temp_version[0]
        if version not in [2, GGUF_VERSION]:
            raise ValueError(f'Unsupported GGUF version {version}')

        if sys.byteorder == "little":
            self.endianess = GGUFEndian.LITTLE if self.byte_order == 'I' else GGUFEndian.BIG
        else:
            self.endianess = GGUFEndian.BIG if self.byte_order == 'I' else GGUFEndian.LITTLE

        self.fields: OrderedDict[str, ReaderField] = OrderedDict()
        self.tensors: list[ReaderTensor] = []
        offs += self._push_field(ReaderField(offs, 'GGUF.version', [temp_version], [0], [GGUFValueType.UINT32]))

        temp_counts = self._get(offs, np.uint64, 2)
        offs += self._push_field(ReaderField(offs, 'GGUF.tensor_count', [temp_counts[:1]], [0], [GGUFValueType.UINT64]))
        offs += self._push_field(ReaderField(offs, 'GGUF.kv_count', [temp_counts[1:]], [0], [GGUFValueType.UINT64]))
        tensor_count, kv_count = temp_counts
        offs = self._build_fields(offs, kv_count)

        offs, tensors_fields = self._build_tensor_info(offs, tensor_count)
        new_align = self.fields.get('general.alignment')
        if new_align is not None:
            self.alignment = int(new_align.parts[-1][0])
        padding = offs % self.alignment
        if padding != 0:
            offs += self.alignment - padding
        self.data_offset = offs
        self._build_tensors(offs, tensors_fields)

    def get_field(self, key: str) -> ReaderField | None:
        return self.fields.get(key, None)

    def get_tensor(self, idx: int) -> ReaderTensor:
        return self.tensors[idx]

    def _get(self, offset: int, dtype, count: int = 1, override_order: str | None = None) -> np.ndarray:
        count = int(count)
        itemsize = int(np.empty([], dtype=dtype).itemsize)
        end_offs = offset + itemsize * count
        arr = self.data[offset:end_offs].view(dtype=dtype)[:count]
        return arr.view(arr.dtype.newbyteorder(self.byte_order if override_order is None else override_order))

    def _push_field(self, field: ReaderField, skip_sum: bool = False) -> int:
        if field.name in self.fields:
            raise KeyError(f'Duplicate {field.name} at offset {field.offset}')
        self.fields[field.name] = field
        return 0 if skip_sum else sum(int(part.nbytes) for part in field.parts)

    def _get_str(self, offset: int) -> tuple[np.ndarray, np.ndarray]:
        slen_arr = self._get(offset, np.uint64)
        slen = int(slen_arr.flat[0]) if slen_arr.size > 0 else 0
        return slen_arr, self._get(offset + 8, np.uint8, slen)

    def _get_field_parts(self, orig_offs: int, raw_type: int) -> tuple[int, list, list, list]:
        offs = orig_offs
        types: list[GGUFValueType] = []
        gtype = GGUFValueType(raw_type)
        types.append(gtype)
        if gtype == GGUFValueType.STRING:
            sparts = list(self._get_str(offs))
            return sum(int(p.nbytes) for p in sparts), sparts, [1], types
        nptype = self.gguf_scalar_to_np.get(gtype)
        if nptype is not None:
            val = self._get(offs, nptype)
            return int(val.nbytes), [val], [0], types
        if gtype == GGUFValueType.ARRAY:
            raw_itype_arr = self._get(offs, np.uint32)
            raw_itype = int(raw_itype_arr.flat[0])
            offs += int(raw_itype_arr.nbytes)
            alen_arr = self._get(offs, np.uint64)
            alen = int(alen_arr.flat[0])
            offs += int(alen_arr.nbytes)
            aparts: list = [raw_itype_arr, alen_arr]
            data_idxs: list[int] = []
            for idx in range(alen):
                curr_size, curr_parts, curr_idxs, curr_types = self._get_field_parts(offs, raw_itype)
                if idx == 0:
                    types += curr_types
                idxs_offs = len(aparts)
                aparts += curr_parts
                data_idxs += (idx + idxs_offs for idx in curr_idxs)
                offs += curr_size
            return offs - orig_offs, aparts, data_idxs, types
        raise ValueError(f'Unknown field type {gtype}')

    def _get_tensor_info_field(self, orig_offs: int) -> ReaderField:
        offs = orig_offs
        name_len, name_data = self._get_str(offs)
        offs += int(name_len.nbytes + name_data.nbytes)
        n_dims = self._get(offs, np.uint32)
        offs += int(n_dims.nbytes)
        dims = self._get(offs, np.uint64, n_dims[0])
        offs += int(dims.nbytes)
        raw_dtype = self._get(offs, np.uint32)
        offs += int(raw_dtype.nbytes)
        offset_tensor = self._get(offs, np.uint64)
        offs += int(offset_tensor.nbytes)
        return ReaderField(orig_offs, str(bytes(name_data), encoding='utf-8'),
                          [name_len, name_data, n_dims, dims, raw_dtype, offset_tensor], [1, 3, 4, 5])

    def _build_fields(self, offs: int, count: int) -> int:
        for _ in range(count):
            orig_offs = offs
            kv_klen, kv_kdata = self._get_str(offs)
            kv_kdata_bytes = bytes(kv_kdata)
            offs += int(kv_klen.nbytes + kv_kdata.nbytes)
            raw_kv_type_arr = self._get(offs, np.uint32)
            offs += int(raw_kv_type_arr.nbytes)
            parts = [kv_klen, kv_kdata, raw_kv_type_arr]
            idxs_offs = len(parts)
            # Handle both scalar and array raw_kv_type
            if raw_kv_type_arr.size > 0:
                rv = int(raw_kv_type_arr.flat[0])
            else:
                rv = 0
            field_size, field_parts, field_idxs, field_types = self._get_field_parts(offs, rv)
            parts += field_parts
            try:
                field_name = kv_kdata_bytes.decode('utf-8')
            except UnicodeDecodeError:
                field_name = kv_kdata_bytes.decode('utf-8', errors='replace')
            self._push_field(ReaderField(orig_offs, field_name,
                                        parts, [idx + idxs_offs for idx in field_idxs], field_types), skip_sum=True)
            offs += field_size
        return offs

    def _build_tensor_info(self, offs: int, count: int) -> tuple[int, list]:
        tensor_fields = []
        for _ in range(count):
            field = self._get_tensor_info_field(offs)
            offs += sum(int(part.nbytes) for part in field.parts)
            tensor_fields.append(field)
        return offs, tensor_fields

    def _build_tensors(self, start_offs: int, fields: list) -> None:
        tensors = []
        tensor_names = set()
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            tensor_name = str(bytes(name_data), encoding='utf-8')
            if tensor_name in tensor_names:
                raise ValueError(f'Duplicate tensor: {tensor_name}')
            tensor_names.add(tensor_name)
            ggml_type = GGMLQuantizationType(raw_dtype[0])
            n_elems = int(np.prod(dims))
            np_dims = tuple(reversed(dims.tolist()))
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            data_offs = int(start_offs + offset_tensor[0])

            if ggml_type == GGMLQuantizationType.F32:
                item_count, item_type = n_elems, np.float32
            elif ggml_type == GGMLQuantizationType.F16:
                item_count, item_type = n_elems, np.float16
            elif ggml_type == GGMLQuantizationType.F64:
                item_count, item_type = n_elems, np.float64
            elif ggml_type == GGMLQuantizationType.I8:
                item_count, item_type = n_elems, np.int8
            elif ggml_type == GGMLQuantizationType.I16:
                item_count, item_type = n_elems, np.int16
            elif ggml_type == GGMLQuantizationType.I32:
                item_count, item_type = n_elems, np.int32
            elif ggml_type == GGMLQuantizationType.I64:
                item_count, item_type = n_elems, np.int64
            else:
                item_count, item_type = n_bytes, np.uint8
                np_dims = quant_shape_to_byte_shape(np_dims, ggml_type)

            tensors.append(ReaderTensor(
                name=tensor_name, tensor_type=ggml_type, shape=dims,
                n_elements=n_elems, n_bytes=n_bytes, data_offset=data_offs,
                data=self._get(data_offs, item_type, item_count).reshape(np_dims), field=field,
            ))
        self.tensors = tensors


# ============================================================
# Tensor name mapping (GGUF -> HuggingFace)
# ============================================================

GGUF_TO_HF_TENSOR_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "token_embd_norm.weight": "model.embed_tokens.norm.weight",
    # Vision encoder
    "v.patch_embed.weight": "vision_tower.patch_embed.weight",
    "v.patch_embed.bias": "vision_tower.patch_embed.bias",
    "v.pos_embed.weight": "vision_tower.pos_embed.weight",
    "v.merger.linear_fc1.weight": "vision_tower.merger.linear_fc1.weight",
    "v.merger.linear_fc1.bias": "vision_tower.merger.linear_fc1.bias",
    "v.merger.linear_fc2.weight": "vision_tower.merger.linear_fc2.weight",
    "v.merger.linear_fc2.bias": "vision_tower.merger.linear_fc2.bias",
    "v.merger.norm.weight": "vision_tower.merger.norm.weight",
    "v.merger.norm.bias": "vision_tower.merger.norm.bias",
    # MTP
    "mtp.norm.weight": "model.layers.mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight": "model.layers.mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight": "model.layers.mtp.pre_fc_norm_hidden.weight",
}

GGUF_TO_HF_BLOCK_MAP = {
    # Attention
    "attn_norm.weight": "input_layernorm.weight",
    "attn_norm.bias": "input_layernorm.bias",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.bias": "post_attention_layernorm.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_out.weight": "self_attn.o_proj.weight",
    "attn_out.bias": "self_attn.o_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_output.bias": "self_attn.o_proj.bias",
    "attn_qkv.weight": "self_attn.qkv_proj.weight",
    "attn_qkv.bias": "self_attn.qkv_proj.bias",
    "attn_gate.weight": "self_attn.gate_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # FFN
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_gate.bias": "mlp.gate_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_down.bias": "mlp.down_proj.bias",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_gate_up.weight": "mlp.gate_up_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    # Post attention norm (Qwen3.5 specific)
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    # SSM/Mamba
    "ssm_a": "mamba.A_log",
    "ssm_dt": "mamba.dt_proj.bias",
    "ssm_norm.weight": "mamba.norm.weight",
    "ssm_conv1d.weight": "mamba.conv1d.weight",
    "ssm_conv1d.bias": "mamba.conv1d.bias",
    "ssm_alpha.weight": "mamba.alpha.weight",
    "ssm_beta.weight": "mamba.beta.weight",
    "ssm_out.weight": "mamba.out_proj.weight",
    "ssm_out.bias": "mamba.out_proj.bias",
    # Vision block
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_out.bias": "self_attn.out_proj.bias",
    "norm1.weight": "layer_norm1.weight",
    "norm1.bias": "layer_norm1.bias",
    "norm2.weight": "layer_norm2.weight",
    "norm2.bias": "layer_norm2.bias",
    "mlp.linear_fc1.weight": "mlp.fc1.weight",
    "mlp.linear_fc1.bias": "mlp.fc1.bias",
    "mlp.linear_fc2.weight": "mlp.fc2.weight",
    "mlp.linear_fc2.bias": "mlp.fc2.bias",
}

GGUF_TO_HF_VISION_BLOCK_MAP = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_out.bias": "self_attn.out_proj.bias",
    "norm1.weight": "layer_norm1.weight",
    "norm1.bias": "layer_norm1.bias",
    "norm2.weight": "layer_norm2.weight",
    "norm2.bias": "layer_norm2.bias",
    "mlp.linear_fc1.weight": "mlp.fc1.weight",
    "mlp.linear_fc1.bias": "mlp.fc1.bias",
    "mlp.linear_fc2.weight": "mlp.fc2.weight",
    "mlp.linear_fc2.bias": "mlp.fc2.bias",
}

GGUF_TO_HF_EXPERT_MAP = {
    "ffn_gate_exp.weight": "mlp.experts.{expert_id}.gate_proj.weight",
    "ffn_down_exp.weight": "mlp.experts.{expert_id}.down_proj.weight",
    "ffn_up_exp.weight": "mlp.experts.{expert_id}.up_proj.weight",
    "ffn_gate_up_exp.weight": "mlp.experts.{expert_id}.gate_up_proj.weight",
}


def gguf_tensor_name_to_hf(name: str, arch: str) -> str | None:
    for gguf_name, hf_name in GGUF_TO_HF_TENSOR_MAP.items():
        if name == gguf_name:
            return hf_name

    if name.startswith("blk."):
        parts = name.split(".")
        if len(parts) >= 3:
            bid = parts[1]
            tensor_name = ".".join(parts[2:])

            for gguf_suffix, hf_suffix in GGUF_TO_HF_EXPERT_MAP.items():
                if tensor_name.endswith(gguf_suffix):
                    hf_base = hf_suffix.replace(".{expert_id}", "_exps")
                    return f"model.layers.{bid}.{hf_base}"

            hf_base = GGUF_TO_HF_BLOCK_MAP.get(tensor_name)
            if hf_base is not None:
                return f"model.layers.{bid}.{hf_base}"

    # Vision encoder: v.blk.{N}.{tensor}
    if name.startswith("v.blk."):
        parts = name.split(".")
        if len(parts) >= 4:
            bid = parts[2]
            tensor_name = ".".join(parts[3:])
            hf_base = GGUF_TO_HF_VISION_BLOCK_MAP.get(tensor_name)
            if hf_base is not None:
                return f"vision_tower.blocks.{bid}.{hf_base}"

    # MTP layers: mtp.layers.{N}.{tensor}
    if name.startswith("mtp.layers."):
        parts = name.split(".")
        if len(parts) >= 4:
            bid = parts[2]
            tensor_name = ".".join(parts[3:])
            hf_base = GGUF_TO_HF_BLOCK_MAP.get(tensor_name)
            if hf_base is not None:
                return f"model.mtp.layers.{bid}.{hf_base}"

    return None


# ============================================================
# RoPE permutation undo
# ============================================================

def undo_rope_permute(weights: np.ndarray, n_head: int, n_kv_head: int | None = None) -> np.ndarray:
    """Reverse the RoPE permutation applied during HF->GGUF conversion."""
    if n_kv_head is not None and n_head != n_kv_head and n_kv_head > 0:
        n_head = n_head // n_kv_head

    dim = weights.shape[0]

    # Check if the reshape is valid
    if n_head <= 0 or dim % (n_head * 2) != 0:
        # Tensor shape doesn't match expected pattern, return as-is
        return weights

    head_dim = dim // (n_head * 2)
    if head_dim <= 0:
        return weights

    remaining_shape = weights.shape[1:]
    weights = weights.reshape(n_head, 2, head_dim, *remaining_shape)
    weights = np.swapaxes(weights, 1, 2)
    weights = weights.reshape(dim, *remaining_shape)
    return weights


# ============================================================
# Config building
# ============================================================

def build_hf_config(reader: GGUFReader, arch: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "architectures": [f"{arch}ForCausalLM"],
        "model_type": arch.lower(),
        "torch_dtype": "float32",
    }

    def get_field(key: str, default: Any = None) -> Any:
        field = reader.get_field(key)
        if field is None:
            return default
        return field.contents()

    for gguf_key, hf_key in [
        ("vocab_size", "vocab_size"),
        ("embedding_length", "hidden_size"),
        ("feed_forward_length", "intermediate_size"),
        ("block_count", "num_hidden_layers"),
        ("attention.head_count", "num_attention_heads"),
        ("attention.head_count_kv", "num_key_value_heads"),
        ("attention.layer_norm_rms_epsilon", "rms_norm_eps"),
        ("attention.layer_norm_epsilon", "layer_norm_eps"),
        ("rope.dimension_count", "head_dim"),
        ("rope.freq_base", "rope_theta"),
        ("context_length", "max_position_embeddings"),
        ("expert_count", "num_local_experts"),
        ("expert_used_count", "num_experts_per_tok"),
        ("attention.sliding_window", "sliding_window"),
    ]:
        val = get_field(f"{arch}.{gguf_key}")
        if val is not None:
            config[hf_key] = int(val) if isinstance(val, (int, float, np.integer)) else val

    config.setdefault("hidden_act", "silu")
    config.setdefault("initializer_range", 0.02)
    config.setdefault("use_cache", True)

    bos = get_field("tokenizer.ggml.bos_token_id")
    if bos is not None:
        config["bos_token_id"] = int(bos)
    eos = get_field("tokenizer.ggml.eos_token_id")
    if eos is not None:
        config["eos_token_id"] = int(eos)

    return {k: v for k, v in config.items() if v is not None}


def split_expert_tensor(data: np.ndarray, tensor_name: str, n_experts: int) -> list[tuple[str, np.ndarray]]:
    if data.ndim < 3:
        return [(tensor_name, data)]
    actual_experts = data.shape[0]
    results = []
    for expert_id in range(actual_experts):
        expert_data = np.ascontiguousarray(data[expert_id])
        hf_name = tensor_name.replace("_exps", f"experts.{expert_id}")
        results.append((hf_name, expert_data))
    return results


# ============================================================
# Main conversion
# ============================================================

def convert_gguf_to_hf(input_path: str | Path, output_dir: str | Path,
                       outtype: str = "f32", split_experts: bool = True,
                       undo_permute: bool = True, use_safetensors: bool = True):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading GGUF: {input_path}")
    reader = GGUFReader(input_path, "r")

    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        raise ValueError("No architecture found in GGUF")
    arch = arch_field.contents()
    logger.info(f"Architecture: {arch}")

    config = build_hf_config(reader, arch)

    out_dtype = {"f32": np.float32, "f16": np.float16, "bf16": np.float32}[outtype]

    n_head = config.get("num_attention_heads", 0)
    n_kv_head_raw = config.get("num_key_value_heads", n_head)
    # Handle list values (e.g., mrope sections in Qwen3.5)
    if isinstance(n_kv_head_raw, list):
        n_kv_head = sum(n_kv_head_raw)
    else:
        n_kv_head = n_kv_head_raw
    n_experts = config.get("num_local_experts", 0)

    state_dict: dict[str, np.ndarray] = {}
    skipped = []

    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        ggml_type = tensor.tensor_type

        logger.info(f"Processing: {name} ({ggml_type.name}, {data.shape})")

        # Get logical shape (before byte conversion for quantized tensors)
        # GGUF stores dims in GGML order (reversed), GGUFReader reverses them back
        # For quantized tensors, GGUFReader converts to byte shape
        if ggml_type not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.F64,
                             GGMLQuantizationType.I8, GGMLQuantizationType.I16, GGMLQuantizationType.I32, GGMLQuantizationType.I64):
            # Quantized tensor: get logical shape from tensor.shape (GGML order, reversed)
            logical_shape = tuple(reversed(tensor.shape.tolist()))
        else:
            logical_shape = data.shape

        # Dequantize if needed
        if ggml_type not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.F64):
            logger.info(f"  Dequantizing from {ggml_type.name}")
            data = dequantize(data, ggml_type)
            # Reshape to logical shape
            n_elements = int(np.prod(logical_shape))
            if data.size == n_elements:
                data = data.reshape(logical_shape)

        # Convert dtype
        if out_dtype == np.float16 and data.dtype != np.float16:
            data = data.astype(np.float16)
        elif out_dtype == np.float32 and data.dtype != np.float32:
            data = data.astype(np.float32)

        # GGUFReader already reverses dims, so no transpose needed

        # Undo RoPE permute
        if undo_permute and n_head > 0:
            if name.endswith(".attn_q.weight"):
                logger.info("  Undoing RoPE permute (Q)")
                data = undo_rope_permute(data, n_head, n_head)
            elif name.endswith(".attn_k.weight"):
                logger.info("  Undoing RoPE permute (K)")
                data = undo_rope_permute(data, n_head, n_kv_head)

        # Map name
        hf_name = gguf_tensor_name_to_hf(name, arch)
        if hf_name is None:
            logger.warning(f"  Cannot map: {name}")
            skipped.append(name)
            continue

        # Split experts
        if split_experts and n_experts > 0 and "_exps" in hf_name and data.ndim >= 3:
            logger.info(f"  Splitting {n_experts} experts")
            for ename, edata in split_expert_tensor(data, hf_name, n_experts):
                state_dict[ename] = edata
        else:
            state_dict[hf_name] = data

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config.json")

    # Save tokenizer files
    tokens_field = reader.get_field("tokenizer.ggml.tokens")
    if tokens_field is not None:
        tokens = tokens_field.contents()
        if isinstance(tokens, list):
            with open(output_dir / "tokenizer.model.tokens", "w") as f:
                for t in tokens:
                    f.write(f"{t}\n")

    bos = reader.get_field("tokenizer.ggml.bos_token_id")
    eos = reader.get_field("tokenizer.ggml.eos_token_id")
    special_tokens = {}
    if bos is not None:
        special_tokens["bos_token"] = {"id": int(bos.contents())}
    if eos is not None:
        special_tokens["eos_token"] = {"id": int(eos.contents())}
    if special_tokens:
        with open(output_dir / "special_tokens_map.json", "w") as f:
            json.dump(special_tokens, f, indent=2)

    # Save model
    try:
        from safetensors.torch import save_file
        has_safetensors = True
    except ImportError:
        has_safetensors = False

    if use_safetensors and has_safetensors:
        import torch
        torch_state = {}
        for n, d in state_dict.items():
            torch_state[n] = torch.from_numpy(d.copy())
        total = sum(t.numel() * t.element_size() for t in torch_state.values())
        if total <= 5 * 1024**3:
            save_file(torch_state, output_dir / "model.safetensors")
            logger.info(f"Saved model.safetensors ({total/1e9:.2f} GB)")
        else:
            logger.warning("Model > 5GB, sharding not yet implemented, saving as single file")
            save_file(torch_state, output_dir / "model.safetensors")
    else:
        try:
            import torch
            torch_state = {}
            for n, d in state_dict.items():
                torch_state[n] = torch.from_numpy(d.copy())
            torch.save(torch_state, output_dir / "pytorch_model.bin")
            logger.info("Saved pytorch_model.bin")
        except ImportError:
            logger.error("Install torch or safetensors: pip install torch safetensors")
            sys.exit(1)

    # generation_config
    gen = {"_from_model_config": True}
    if config.get("bos_token_id") is not None:
        gen["bos_token_id"] = config["bos_token_id"]
    if config.get("eos_token_id") is not None:
        gen["eos_token_id"] = config["eos_token_id"]
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen, f, indent=2)

    logger.info(f"Done! {len(state_dict)} tensors, {len(skipped)} skipped")
    if skipped:
        logger.info(f"Skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")


def main():
    parser = argparse.ArgumentParser(description="GGUF -> HuggingFace converter (standalone)")
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--outtype", "-t", default="f32", choices=["f32", "f16", "bf16"])
    parser.add_argument("--no-split-experts", action="store_true")
    parser.add_argument("--no-undo-permute", action="store_true")
    parser.add_argument("--no-safetensors", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(message)s")

    if not Path(args.input).exists():
        logger.error(f"File not found: {args.input}")
        sys.exit(1)

    convert_gguf_to_hf(
        input_path=args.input, output_dir=args.output_dir,
        outtype=args.outtype, split_experts=not args.no_split_experts,
        undo_permute=not args.no_undo_permute, use_safetensors=not args.no_safetensors,
    )


if __name__ == "__main__":
    main()
