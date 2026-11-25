import sys
from pathlib import Path

from utils import compile_npu_kernel
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


if __name__ == "__main__":
    signature = {
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "output_ptr": "*fp32",
        "n_elements": "i32",
    }

    constexprs = {"BLOCK_SIZE": 64}

    out_dir = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "output"
    compile_npu_kernel(add_kernel, signature, constexprs, "add_kernel", out_dir)
