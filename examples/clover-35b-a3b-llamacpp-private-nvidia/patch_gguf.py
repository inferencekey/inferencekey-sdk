"""Make the official Clover-35B-A3B-Preview GGUF load in upstream llama.cpp (v0.9.0).

Why: the GGUF files DiffuSpace published on 2026-10-28 were written by their own
llama.cpp build (the one with self-speculative decoding). The Clover support
that upstream merged on 2026-11-02 (ggml-org/llama.cpp#31306, first release
v0.9.0) is autoregressive only and expects a slightly different file:

  1. it requires the key `clover_moe.expert_shared_count` (Clover has one
     shared expert per layer; the official file does not write the key), and
     llama-server stops with
     "key not found in model: clover_moe.expert_shared_count";
  2. its converter drops the diffusion-only tensors (`ssm_bd_history.*`,
     `ssm_bd_logit_scale.weight`, `ssm_conv1d.weight_right`, 4 per linear
     attention layer, 120 in total), so the loader would also stop with
     "wrong number of tensors";
  3. it expects `blk.N.ssm_a` to hold -exp(A_log), as for Qwen3.5/3.6, while
     the official file stores the raw A_log. Loaded as is, the model runs and
     answers with garbage (we got 600 x "G" for "What is the capital of
     Australia?").

This script copies the GGUF, adds the key, leaves those 120 tensors out and
rewrites the 30 small `ssm_a` vectors as -exp(A_log). Every other tensor is
copied byte for byte (no re-quantization). The AR path of llama.cpp never
reads the dropped tensors.

Usage (needs the `gguf` package from llama.cpp's gguf-py, `pip install gguf`
or PYTHONPATH=<llama.cpp>/gguf-py):
    python patch_gguf.py Clover-35B-A3B-Preview-Q4_0.gguf Clover-35B-A3B-Preview-Q4_0-ar.gguf
"""

from __future__ import annotations

import sys

import gguf
import numpy as np

DIFFUSION_ONLY = ("ssm_bd_history.weight", "ssm_bd_history.bias", "ssm_bd_logit_scale.weight", "ssm_conv1d.weight_right")


def main(src: str, dst: str) -> None:
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    key = f"{arch}.expert_shared_count"
    if key not in reader.fields:
        writer.add_key_value(key, 1, gguf.GGUFValueType.UINT32)

    keep = [t for t in reader.tensors if not t.name.endswith(DIFFUSION_ONLY)]
    print(f"tensors: {len(reader.tensors)} -> {len(keep)} (dropped {len(reader.tensors) - len(keep)} diffusion-only)")
    for t in keep:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    fixed = 0
    for t in keep:
        data = t.data
        if t.name.endswith(".ssm_a") and data.dtype == np.float32 and float(np.max(data)) > 0:
            data = (-np.exp(data.astype(np.float64))).astype(np.float32)  # raw A_log -> -exp(A_log)
            fixed += 1
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    print(f"ssm_a tensors rewritten as -exp(A_log): {fixed}")
    writer.close()
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
