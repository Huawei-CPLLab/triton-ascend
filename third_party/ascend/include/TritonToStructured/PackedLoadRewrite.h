#ifndef TRITON_ASCEND_PACKED_LOAD_REWRITE_H
#define TRITON_ASCEND_PACKED_LOAD_REWRITE_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

struct PackedLoadRewriteState {
  llvm::DenseMap<std::pair<Value, int64_t>, Value> compactLoads;
};

// Packed quantized formats may express logical expansion in pointer
// arithmetic. On Ascend that can turn replication into scalar GM accesses.
// This rewrite is deliberately limited to statically provable offset maps.
class PackedLoadRewrite : public OpRewritePattern<LoadOp> {
public:
  PackedLoadRewrite(MLIRContext *context, PackedLoadRewriteState *state,
                   PatternBenefit benefit = 1)
      : OpRewritePattern<LoadOp>(context, benefit), state(state) {}

  LogicalResult matchAndRewrite(LoadOp op,
                                PatternRewriter &rewriter) const override;

private:
  PackedLoadRewriteState *state;
};

} // namespace mlir::triton

#endif
