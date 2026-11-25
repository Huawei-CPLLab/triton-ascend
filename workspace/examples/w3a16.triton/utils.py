import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import triton
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.ascend.compiler import (
    AscendBackend,
    make_ttir,
    ttir_to_linalg,
    linalg_to_bin_enable_npu_compile_910_95,
)
from triton.compiler import ASTSource
from triton.backends.ascend.driver import NPUUtils


def compile_npu_kernel(kernel_fn, signature, constexprs, kernel_name, out_dir=None):
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "output"
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    os.environ["WS_KERNEL_OUTPUT_DIR"] = str(out_path)
    os.environ["WS_KERNEL_NAME"] = kernel_name

    target = GPUTarget("npu", "Kirin9020", 32)
    backend = AscendBackend(target)

    NPUUtils.has_device_limit = lambda self: False

    options = backend.parse_options({
        "debug": True,
        "compile_mode": "simd",
        "enable_dynamic_cv_pipeline": False,
        "enable_auto_bind_sub_block": False,
        "enable_auto_multi_buffer": True,
        "limit_auto_multi_buffer_buffer": "no-limit",
        "multibuffer": True,
        "num_stages": 2,
    })

    source = ASTSource(
        fn=kernel_fn,
        signature=signature,
        constexprs=constexprs,
    )

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    metadata = {
        "hash": source.hash(),
        "target": target,
        **options.__dict__,
    }

    module = source.make_ir(
        target=target,
        options=options,
        codegen_fns=backend.get_codegen_implementation(options),
        module_map=backend.get_module_map(),
        context=context,
    )

    source_path = out_path / f"01-{kernel_name}.source.ttir.mlir"
    source_path.write_text(str(module))

    ttir = make_ttir(
        module,
        metadata,
        options,
    )

    ttir_path = out_path / f"02-{kernel_name}.optimized.ttir.mlir"
    ttir_path.write_text(str(ttir))

    ttadapter = ttir_to_linalg(
        ttir,
        metadata,
        options,
        named_ops=True,
    )

    ttadapter_path = out_path / f"03-{kernel_name}.ttadapter.mlir"
    ttadapter_path.write_text(str(ttadapter))

    npubin = linalg_to_bin_enable_npu_compile_910_95(
        str(ttadapter),
        metadata,
        options,
    )

    npubin_path = out_path / f"06-{kernel_name}.o"
    npubin_path.write_bytes(npubin)
