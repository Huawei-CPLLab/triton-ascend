#include "TritonToStructured/PackedLoadRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::triton;

namespace {
// The packed-load rewrite handles logical 2-D tensors stored in compressed
// sub-byte memory layouts for quantized weights (such as W1, W2, W3-QH, W3-QS,
// W4, etc.). The rewrite does not change the kernel's meaning; it recognizes
// the packed offset pattern, loads the compact physical buffer once, and
// reshapes/broadcasts the values back to the logical tensor shape.
struct StaticTensor {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> values;
};

// Small constant-folding evaluator used to recognize statically known pointer
// arithmetic. This function accepts tensor-producing ops that appear in the
// packed qweight offset patterns.
static FailureOr<StaticTensor> evaluate(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isIntOrIndex())
    return failure();
  StaticTensor result;
  result.shape.assign(type.getShape().begin(), type.getShape().end());
  result.values.resize(type.getNumElements());

  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    if (result.shape.size() != 1 || range.getStart() < 0 ||
        range.getEnd() - range.getStart() != result.shape[0])
      return failure();
    for (int64_t i = 0; i < result.shape[0]; ++i)
      result.values[i] = range.getStart() + i;
    return result;
  }

  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseIntOrFPElementsAttr>(constant.getValue());
    if (!dense || dense.getNumElements() != static_cast<int64_t>(result.values.size()))
      return failure();
    for (auto it : llvm::enumerate(dense.getValues<APInt>()))
      result.values[it.index()] = it.value().getSExtValue();
    return result;
  }

  if (auto splat = value.getDefiningOp<SplatOp>()) {
    auto scalar = splat.getSrc();
    auto scalarConst = scalar.getDefiningOp<arith::ConstantIntOp>();
    if (!scalarConst)
      return failure();
    std::fill(result.values.begin(), result.values.end(), scalarConst.value());
    return result;
  }

  if (auto expand = value.getDefiningOp<ExpandDimsOp>()) {
    auto source = evaluate(expand.getSrc());
    if (failed(source))
      return failure();
    auto axis = expand.getAxis();
    if (axis < 0 || axis > static_cast<int64_t>(source->shape.size()))
      return failure();
    SmallVector<int64_t> expected;
    expected.append(source->shape.begin(), source->shape.begin() + axis);
    expected.push_back(1);
    expected.append(source->shape.begin() + axis, source->shape.end());
    if (expected != result.shape)
      return failure();
    result.values = source->values;
    return result;
  }

  if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    auto source = evaluate(broadcast.getSrc());
    if (failed(source) || source->shape.size() > result.shape.size())
      return failure();
    size_t leading = result.shape.size() - source->shape.size();
    for (size_t i = 0; i < source->shape.size(); ++i) {
      if (source->shape[i] != 1 && source->shape[i] != result.shape[leading + i])
        return failure();
    }
    SmallVector<int64_t> strides(result.shape.size(), 1);
    for (int64_t i = result.shape.size() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * result.shape[i + 1];
    SmallVector<int64_t> sourceStrides(source->shape.size(), 1);
    for (int64_t i = source->shape.size() - 2; i >= 0; --i)
      sourceStrides[i] = sourceStrides[i + 1] * source->shape[i + 1];
    for (int64_t linear = 0; linear < static_cast<int64_t>(result.values.size()); ++linear) {
      int64_t sourceLinear = 0;
      int64_t remainder = linear;
      for (size_t i = 0; i < result.shape.size(); ++i) {
        int64_t coord = remainder / strides[i];
        remainder %= strides[i];
        if (i >= leading && source->shape[i - leading] != 1)
          sourceLinear += coord * sourceStrides[i - leading];
      }
      result.values[linear] = source->values[sourceLinear];
    }
    return result;
  }

  auto binary = [&](Value lhs, Value rhs, auto operation) -> FailureOr<StaticTensor> {
    auto left = evaluate(lhs);
    auto right = evaluate(rhs);
    if (failed(left) || failed(right) || left->shape != right->shape)
      return failure();
    StaticTensor combined = *left;
    for (size_t i = 0; i < combined.values.size(); ++i)
      combined.values[i] = operation(left->values[i], right->values[i]);
    return combined;
  };
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return binary(add.getLhs(), add.getRhs(), [](int64_t a, int64_t b) { return a + b; });
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return binary(sub.getLhs(), sub.getRhs(), [](int64_t a, int64_t b) { return a - b; });
  if (auto mul = value.getDefiningOp<arith::MulIOp>())
    return binary(mul.getLhs(), mul.getRhs(), [](int64_t a, int64_t b) { return a * b; });
  if (auto band = value.getDefiningOp<arith::AndIOp>())
    return binary(band.getLhs(), band.getRhs(), [](int64_t a, int64_t b) { return a & b; });
  if (auto bor = value.getDefiningOp<arith::OrIOp>())
    return binary(bor.getLhs(), bor.getRhs(), [](int64_t a, int64_t b) { return a | b; });
  if (auto bxor = value.getDefiningOp<arith::XOrIOp>())
    return binary(bxor.getLhs(), bxor.getRhs(), [](int64_t a, int64_t b) { return a ^ b; });
  if (auto shr = value.getDefiningOp<arith::ShRSIOp>())
    return binary(shr.getLhs(), shr.getRhs(), [](int64_t a, int64_t b) { return a >> b; });
  if (auto shru = value.getDefiningOp<arith::ShRUIOp>())
    return binary(shru.getLhs(), shru.getRhs(), [](int64_t a, int64_t b) {
      return static_cast<int64_t>(static_cast<uint64_t>(a) >> b);
    });
  if (auto shl = value.getDefiningOp<arith::ShLIOp>())
    return binary(shl.getLhs(), shl.getRhs(), [](int64_t a, int64_t b) { return a << b; });
  if (auto divs = value.getDefiningOp<arith::DivSIOp>())
    return binary(divs.getLhs(), divs.getRhs(), [](int64_t a, int64_t b) {
      return b != 0 ? a / b : 0;
    });
  if (auto divu = value.getDefiningOp<arith::DivUIOp>())
    return binary(divu.getLhs(), divu.getRhs(), [](int64_t a, int64_t b) {
      return b != 0 ? static_cast<int64_t>(static_cast<uint64_t>(a) / static_cast<uint64_t>(b)) : 0;
    });
  if (auto rems = value.getDefiningOp<arith::RemSIOp>())
    return binary(rems.getLhs(), rems.getRhs(), [](int64_t a, int64_t b) {
      return b != 0 ? a % b : 0;
    });
  if (auto remu = value.getDefiningOp<arith::RemUIOp>())
    return binary(remu.getLhs(), remu.getRhs(), [](int64_t a, int64_t b) {
      return b != 0 ? static_cast<int64_t>(static_cast<uint64_t>(a) % static_cast<uint64_t>(b)) : 0;
    });
  if (auto cast = value.getDefiningOp<arith::ExtSIOp>())
    return evaluate(cast.getIn());
  if (auto cast = value.getDefiningOp<arith::ExtUIOp>())
    return evaluate(cast.getIn());
  if (auto cast = value.getDefiningOp<arith::TruncIOp>())
    return evaluate(cast.getIn());
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return evaluate(cast.getIn());
  return failure();
}

static bool isAllTrueMask(Value value) {
  auto maskType = dyn_cast<RankedTensorType>(value.getType());
  if (!maskType || !maskType.hasStaticShape() ||
      !maskType.getElementType().isInteger(1))
    return false;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseIntOrFPElementsAttr>(constant.getValue());
    if (!dense)
      return false;
    return llvm::all_of(dense.getValues<APInt>(),
                        [](const APInt &val) { return val.isOne(); });
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return isAllTrueMask(broadcast.getSrc());
  if (auto splat = value.getDefiningOp<SplatOp>()) {
    if (auto constant = splat.getSrc().getDefiningOp<arith::ConstantIntOp>())
      return constant.value() == 1;
    return false;
  }
  if (auto band = value.getDefiningOp<arith::AndIOp>())
    return isAllTrueMask(band.getLhs()) && isAllTrueMask(band.getRhs());

  auto cmp = value.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return false;
  auto lhs = evaluate(cmp.getLhs());
  auto rhs = evaluate(cmp.getRhs());
  if (failed(lhs) || failed(rhs) || lhs->shape != rhs->shape)
    return false;
  auto predicate = cmp.getPredicate();
  for (size_t i = 0; i < lhs->values.size(); ++i) {
    int64_t left = lhs->values[i];
    int64_t right = rhs->values[i];
    bool result = false;
    switch (predicate) {
    case arith::CmpIPredicate::eq: result = left == right; break;
    case arith::CmpIPredicate::ne: result = left != right; break;
    case arith::CmpIPredicate::slt: result = left < right; break;
    case arith::CmpIPredicate::sle: result = left <= right; break;
    case arith::CmpIPredicate::sgt: result = left > right; break;
    case arith::CmpIPredicate::sge: result = left >= right; break;
    case arith::CmpIPredicate::ult: result = static_cast<uint64_t>(left) < static_cast<uint64_t>(right); break;
    case arith::CmpIPredicate::ule: result = static_cast<uint64_t>(left) <= static_cast<uint64_t>(right); break;
    case arith::CmpIPredicate::ugt: result = static_cast<uint64_t>(left) > static_cast<uint64_t>(right); break;
    case arith::CmpIPredicate::uge: result = static_cast<uint64_t>(left) >= static_cast<uint64_t>(right); break;
    }
    if (!result)
      return false;
  }
  return true;
}

static FailureOr<StaticTensor> evaluatePointerOffset(Value value) {
  if (auto addPtr = value.getDefiningOp<AddPtrOp>()) {
    auto ownOffset = evaluate(addPtr.getOffset());
    if (failed(ownOffset))
      return failure();

    Value parentValue = addPtr.getPtr();
    while (auto broadcast = parentValue.getDefiningOp<BroadcastOp>())
      parentValue = broadcast.getSrc();
    auto parent = parentValue.getDefiningOp<AddPtrOp>();
    if (!parent)
      return ownOffset;

    // Only recursively combine if parent is also a tensor AddPtrOp.
    if (!isa<RankedTensorType>(parent.getType()))
      return ownOffset;

    auto parentOffset = evaluatePointerOffset(parent.getResult());
    if (failed(parentOffset))
      return failure();

    if (parentOffset->shape == ownOffset->shape) {
      for (size_t i = 0; i < ownOffset->values.size(); ++i)
        ownOffset->values[i] += parentOffset->values[i];
    } else {
      if (parentOffset->shape.size() != ownOffset->shape.size())
        return failure();
      SmallVector<int64_t> parentStrides(parentOffset->shape.size(), 1);
      SmallVector<int64_t> ownStrides(ownOffset->shape.size(), 1);
      for (int64_t i = parentOffset->shape.size() - 2; i >= 0; --i)
        parentStrides[i] = parentStrides[i + 1] * parentOffset->shape[i + 1];
      for (int64_t i = ownOffset->shape.size() - 2; i >= 0; --i)
        ownStrides[i] = ownStrides[i + 1] * ownOffset->shape[i + 1];
      for (size_t dim = 0; dim < ownOffset->shape.size(); ++dim)
        if (parentOffset->shape[dim] != 1 &&
            parentOffset->shape[dim] != ownOffset->shape[dim])
          return failure();
      for (int64_t linear = 0;
           linear < static_cast<int64_t>(ownOffset->values.size()); ++linear) {
        int64_t remainder = linear;
        int64_t parentLinear = 0;
        for (size_t dim = 0; dim < ownOffset->shape.size(); ++dim) {
          int64_t coordinate = remainder / ownStrides[dim];
          remainder %= ownStrides[dim];
          if (parentOffset->shape[dim] != 1)
            parentLinear += coordinate * parentStrides[dim];
        }
        ownOffset->values[linear] += parentOffset->values[parentLinear];
      }
    }
    return ownOffset;
  }
  return failure();
}

static Value scalarBase(Value value) {
  while (true) {
    if (auto splat = value.getDefiningOp<SplatOp>()) {
      value = splat.getSrc();
      continue;
    }
    if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
      value = broadcast.getSrc();
      continue;
    }
    // Only unwrap tensor AddPtrOp whose offset was folded into tensor offsets.
    if (auto addPtr = value.getDefiningOp<AddPtrOp>()) {
      if (isa<RankedTensorType>(addPtr.getType())) {
        value = addPtr.getPtr();
        continue;
      }
    }
    break;
  }
  return value;
}

enum class PackedPatternKind {
  Consecutive,
  Strided,
  PairSlot
};

struct PackedLayoutDescriptor {
  int64_t groupSize = 0;      // G (e.g., 32, 64, 16, 128)
  int64_t bytesPerGroup = 0;  // P (e.g., 4, 8, 16)
  int64_t repeatCount = 0;    // R = G / P
  PackedPatternKind kind = PackedPatternKind::Consecutive;
  int64_t pairIndex = 0;      // 0 or 1 for PairSlot
  int64_t baseOffset = 0;     // byte offset of (0, 0)
  int64_t rowStride = 0;      // byte stride between successive rows
};

static std::optional<PackedLayoutDescriptor> deducePackedLayout(
    ArrayRef<int64_t> offsets, int64_t rows, int64_t columns) {
  if (offsets.empty() || static_cast<int64_t>(offsets.size()) != rows * columns)
    return std::nullopt;

  int64_t baseOffset = offsets[0];
  int64_t rowStride = (rows > 1) ? (offsets[columns] - offsets[0]) : 0;

  // Verify that all rows follow the same rowStride.
  if (rows > 1) {
    for (int64_t r = 1; r < rows; ++r) {
      if (offsets[r * columns] - offsets[0] != r * rowStride)
        return std::nullopt;
    }
  }

  const int64_t candidateGroups[] = {32, 64, 16, 128};
  for (int64_t G : candidateGroups) {
    if (columns % G != 0)
      continue;
    int64_t numGroups = columns / G;

    for (int64_t P = 1; P < G; P *= 2) {
      int64_t R = G / P;
      int64_t effectiveRowStride = (rows > 1) ? rowStride : (numGroups * P);

      // Pattern 1: Consecutive (Block) Broadcast:
      // expected = baseOffset + r * effectiveRowStride + g * P + (k / R)
      bool matchConsecutive = true;
      for (int64_t r = 0; r < rows && matchConsecutive; ++r) {
        for (int64_t c = 0; c < columns; ++c) {
          int64_t g = c / G;
          int64_t k = c % G;
          int64_t expected = baseOffset + r * effectiveRowStride + g * P + (k / R);
          if (offsets[r * columns + c] != expected) {
            matchConsecutive = false;
            break;
          }
        }
      }
      if (matchConsecutive) {
        PackedLayoutDescriptor desc;
        desc.groupSize = G;
        desc.bytesPerGroup = P;
        desc.repeatCount = R;
        desc.kind = PackedPatternKind::Consecutive;
        desc.baseOffset = baseOffset;
        desc.rowStride = effectiveRowStride;
        return desc;
      }

      // Pattern 2: Strided (Interleaved) Broadcast:
      // expected = baseOffset + r * effectiveRowStride + g * P + (k % P)
      bool matchStrided = true;
      for (int64_t r = 0; r < rows && matchStrided; ++r) {
        for (int64_t c = 0; c < columns; ++c) {
          int64_t g = c / G;
          int64_t k = c % G;
          int64_t expected = baseOffset + r * effectiveRowStride + g * P + (k % P);
          if (offsets[r * columns + c] != expected) {
            matchStrided = false;
            break;
          }
        }
      }
      if (matchStrided) {
        PackedLayoutDescriptor desc;
        desc.groupSize = G;
        desc.bytesPerGroup = P;
        desc.repeatCount = R;
        desc.kind = PackedPatternKind::Strided;
        desc.baseOffset = baseOffset;
        desc.rowStride = effectiveRowStride;
        return desc;
      }

      // Pattern 3: Dual-Load Pair-Slot:
      // expected = pairBaseOffset + r * effectiveRowStride + g * P + (k / (2*R)) * 2 + pair
      if (P >= 2 && 2 * R <= G) {
        int64_t pair = offsets[0] & 1;
        int64_t pairBaseOffset = offsets[0] - pair;
        bool matchPair = true;
        for (int64_t r = 0; r < rows && matchPair; ++r) {
          for (int64_t c = 0; c < columns; ++c) {
            int64_t g = c / G;
            int64_t k = c % G;
            int64_t expected = pairBaseOffset + r * effectiveRowStride + g * P +
                               (k / (2 * R)) * 2 + pair;
            if (offsets[r * columns + c] != expected) {
              matchPair = false;
              break;
            }
          }
        }
        if (matchPair) {
          PackedLayoutDescriptor desc;
          desc.groupSize = G;
          desc.bytesPerGroup = P;
          desc.repeatCount = R;
          desc.kind = PackedPatternKind::PairSlot;
          desc.pairIndex = pair;
          desc.baseOffset = pairBaseOffset;
          desc.rowStride = effectiveRowStride;
          return desc;
        }
      }
    }
  }

  return std::nullopt;
}

static Value createCompactLoad1D(Location loc, Value base, int64_t elements,
                                 int64_t baseOffset,
                                 PatternRewriter &rewriter) {
  auto basePtr = dyn_cast<PointerType>(base.getType());
  if (!basePtr)
    return nullptr;
  auto ptrType = RankedTensorType::get({elements}, basePtr);
  auto indexType = RankedTensorType::get({elements}, rewriter.getI32Type());
  auto range = rewriter.create<MakeRangeOp>(loc, indexType, 0, elements);
  Value offsets = range.getResult();
  if (baseOffset != 0) {
    auto baseOffsetConst =
        rewriter.create<arith::ConstantIntOp>(loc, baseOffset, 32);
    auto baseOffsetSplat =
        rewriter.create<SplatOp>(loc, indexType, baseOffsetConst);
    offsets = rewriter.create<arith::AddIOp>(loc, offsets, baseOffsetSplat);
  }
  auto splat = rewriter.create<SplatOp>(loc, ptrType, base);
  auto ptr = rewriter.create<AddPtrOp>(loc, ptrType, splat, offsets);
  return rewriter
      .create<LoadOp>(loc, ptr.getResult(), nullptr, nullptr,
                      CacheModifier::NONE, EvictionPolicy::NORMAL, false)
      .getResult();
}

static Value createCompactLoad2D(Location loc, Value base, int64_t rows,
                                 int64_t physicalCols, int64_t rowStride,
                                 int64_t baseOffset,
                                 PatternRewriter &rewriter) {
  auto basePtr = dyn_cast<PointerType>(base.getType());
  if (!basePtr)
    return nullptr;

  auto ptrType = RankedTensorType::get({rows, physicalCols}, basePtr);
  auto i32Type = rewriter.getI32Type();
  auto rowType1D = RankedTensorType::get({rows}, i32Type);
  auto colType1D = RankedTensorType::get({physicalCols}, i32Type);
  auto rowType2D = RankedTensorType::get({rows, 1}, i32Type);
  auto colType2D = RankedTensorType::get({1, physicalCols}, i32Type);
  auto full2DIndexType = RankedTensorType::get({rows, physicalCols}, i32Type);

  auto rowRange = rewriter.create<MakeRangeOp>(loc, rowType1D, 0, rows);
  auto rowExpanded = rewriter.create<ExpandDimsOp>(loc, rowType2D, rowRange, 1);
  auto rowStrideConst = rewriter.create<arith::ConstantIntOp>(loc, rowStride, 32);
  auto rowStrideSplat = rewriter.create<SplatOp>(loc, rowType2D, rowStrideConst);
  auto rowOffsets = rewriter.create<arith::MulIOp>(loc, rowExpanded, rowStrideSplat);
  auto rowBroadcast = rewriter.create<BroadcastOp>(loc, full2DIndexType, rowOffsets);

  auto colRange = rewriter.create<MakeRangeOp>(loc, colType1D, 0, physicalCols);
  auto colExpanded = rewriter.create<ExpandDimsOp>(loc, colType2D, colRange, 0);
  auto colBroadcast = rewriter.create<BroadcastOp>(loc, full2DIndexType, colExpanded);

  Value offsets = rewriter.create<arith::AddIOp>(loc, rowBroadcast, colBroadcast);
  if (baseOffset != 0) {
    auto baseOffsetConst =
        rewriter.create<arith::ConstantIntOp>(loc, baseOffset, 32);
    auto baseOffsetSplat =
        rewriter.create<SplatOp>(loc, full2DIndexType, baseOffsetConst);
    offsets = rewriter.create<arith::AddIOp>(loc, offsets, baseOffsetSplat);
  }

  auto splat = rewriter.create<SplatOp>(loc, ptrType, base);
  auto ptr = rewriter.create<AddPtrOp>(loc, ptrType, splat, offsets);
  return rewriter
      .create<LoadOp>(loc, ptr.getResult(), nullptr, nullptr,
                      CacheModifier::NONE, EvictionPolicy::NORMAL, false)
      .getResult();
}
} // namespace

LogicalResult PackedLoadRewrite::matchAndRewrite(
    LoadOp op, PatternRewriter &rewriter) const {
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (op.getMask() && !isAllTrueMask(op.getMask()))
    return failure();
  if (op.getOther() && !op.getMask())
    return failure();
  auto addptr = op.getPtr().getDefiningOp<AddPtrOp>();
  if (!resultType || !resultType.hasStaticShape() || resultType.getRank() != 2)
    return failure();
  if (!addptr)
    return failure();
  auto offsets = evaluatePointerOffset(op.getPtr());
  if (failed(offsets) || offsets->shape != resultType.getShape())
    return failure();

  int64_t rows = resultType.getShape()[0];
  int64_t columns = resultType.getShape()[1];
  auto layout = deducePackedLayout(offsets->values, rows, columns);
  if (!layout)
    return failure();

  int64_t G = layout->groupSize;
  int64_t P = layout->bytesPerGroup;
  int64_t R = layout->repeatCount;
  int64_t groups = columns / G;
  int64_t physicalCols = groups * P;
  int64_t physicalTotal = rows * physicalCols;

  Value base = scalarBase(addptr.getPtr());
  using CompactKey = std::tuple<void *, int64_t, int64_t, int64_t>;
  CompactKey key{base.getAsOpaquePointer(), layout->baseOffset, physicalTotal, layout->rowStride};

  Value compact = Value();
  if (state) {
    auto it = state->compactLoads.find(key);
    if (it != state->compactLoads.end()) {
      Value candidate = it->second;
      if (candidate && candidate.getDefiningOp() &&
          candidate.getDefiningOp()->getBlock() == op->getBlock() &&
          candidate.getDefiningOp()->isBeforeInBlock(op)) {
        compact = candidate;
      }
    }
  }

  if (!compact) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (layout->rowStride == physicalCols && layout->baseOffset == 0) {
      compact = createCompactLoad1D(op.getLoc(), base, physicalTotal, 0, rewriter);
    } else if (layout->rowStride == physicalCols) {
      compact = createCompactLoad1D(op.getLoc(), base, physicalTotal, layout->baseOffset, rewriter);
    } else {
      compact = createCompactLoad2D(op.getLoc(), base, rows, physicalCols,
                                    layout->rowStride, layout->baseOffset, rewriter);
    }
    if (state && compact)
      state->compactLoads[key] = compact;
  }
  if (!compact)
    return failure();

  auto compactType = cast<RankedTensorType>(compact.getType());
  SmallVector<int64_t> packedShape{rows, groups, P};
  auto packed = rewriter.create<ReshapeOp>(
      op.getLoc(), RankedTensorType::get(packedShape, compactType.getElementType()),
      compact);

  int64_t broadcastAxis = (layout->kind == PackedPatternKind::Strided) ? 2 : 3;
  auto expanded = rewriter.create<ExpandDimsOp>(
      op.getLoc(), packed.getResult(), broadcastAxis);
  SmallVector<int64_t> expandedShape = packedShape;
  expandedShape.insert(expandedShape.begin() + broadcastAxis, 1);
  expandedShape[broadcastAxis] = R;
  auto broadcast = rewriter.create<BroadcastOp>(
      op.getLoc(), RankedTensorType::get(expandedShape, compactType.getElementType()),
      expanded.getResult());
  auto restored = rewriter.create<ReshapeOp>(op.getLoc(), resultType,
                                             broadcast.getResult());
  rewriter.replaceOp(op, restored.getResult());
  return success();
}
