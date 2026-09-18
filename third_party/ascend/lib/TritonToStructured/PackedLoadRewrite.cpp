#include "TritonToStructured/PackedLoadRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace {
// The packed-load rewrite handles a very specific class of kernels: logical
// 2-D tensors that are stored in a compressed memory layout for qweight packs
// (W3-QH, W3-QS, and W4).  The rewrite does not change the kernel's meaning;
// it only recognizes the special offset pattern, loads the compact buffer once,
// and reshapes/broadcasts the values back to the logical tensor shape.
struct StaticTensor {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> values;
};

// Small constant-folding evaluator used to recognize statically known pointer
// arithmetic.  This function is intentionally narrow: it accepts only a subset
// of tensor-producing ops that appear in the packed qweight offset pattern.
//
// The reason it exists is simple: before the pass rewrites a load, it must prove
// that the offset expression is a compile-time-known tensor with the exact shape
// required by a packed layout.  Once that proof succeeds, the code can inspect
// the offset table and decide whether the load is W3-QH, W3-QS, or W4.
static FailureOr<StaticTensor> evaluate(Value value) {
  // The evaluator only handles ranked tensors whose shape is fixed at compile
  // time and whose element type is an integer/index.  That is enough for the
  // packed-load offset expressions created by make_range + broadcast + arithmetic.
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isIntOrIndex())
    return failure();
  StaticTensor result;
  result.shape.assign(type.getShape().begin(), type.getShape().end());
  result.values.resize(type.getNumElements());

  // A simple make_range value is one of the easiest cases to evaluate: it is a
  // dense 1-D tensor of successive integers.
  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    if (result.shape.size() != 1 || range.getStart() < 0 ||
        range.getEnd() - range.getStart() != result.shape[0])
      return failure();
    for (int64_t i = 0; i < result.shape[0]; ++i)
      result.values[i] = range.getStart() + i;
    return result;
  }

  // A constant tensor is evaluated element-by-element.  This covers the constant
  // seeds used in the packed offset construction, like splats and simple
  // integer constants that are later combined by arithmetic.
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseIntOrFPElementsAttr>(constant.getValue());
    if (!dense || dense.getNumElements() != result.values.size())
      return failure();
    for (auto it : llvm::enumerate(dense.getValues<APInt>()))
      result.values[it.index()] = it.value().getSExtValue();
    return result;
  }

  // A splat is effectively a scalar broadcast; once the scalar is a constant,
  // every element in the tensor has the same value.
  if (auto splat = value.getDefiningOp<SplatOp>()) {
    auto scalar = splat.getSrc();
    auto scalarConst = scalar.getDefiningOp<arith::ConstantIntOp>();
    if (!scalarConst)
      return failure();
    std::fill(result.values.begin(), result.values.end(),
              scalarConst.value());
    return result;
  }

  // ExpandDims does not change the underlying values; it only inserts a size-1
  // axis.  The evaluator therefore keeps the same flat backing array and checks
  // that the inserted dimension matches the expected shape.
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

  // Broadcast is the main shape-lifting operation for the packed offset pattern.
  // The source is repeated across the leading broadcasted dimensions, and we map
  // the logical linear index back to the source's linear index via stride math.
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

  // Binary operations are the last piece of the offset pattern.  The code below
  // accepts arithmetic used in the packed memory formula, like add, mul, shift,
  // and mask operations, and folds them element-by-element.
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
  if (auto mul = value.getDefiningOp<arith::MulIOp>())
    return binary(mul.getLhs(), mul.getRhs(), [](int64_t a, int64_t b) { return a * b; });
  if (auto band = value.getDefiningOp<arith::AndIOp>())
    return binary(band.getLhs(), band.getRhs(), [](int64_t a, int64_t b) { return a & b; });
  if (auto shr = value.getDefiningOp<arith::ShRSIOp>())
    return binary(shr.getLhs(), shr.getRhs(), [](int64_t a, int64_t b) { return a >> b; });
  if (auto shl = value.getDefiningOp<arith::ShLIOp>())
    return binary(shl.getLhs(), shl.getRhs(), [](int64_t a, int64_t b) { return a << b; });
  if (auto add = value.getDefiningOp<arith::SubIOp>())
    return binary(add.getLhs(), add.getRhs(), [](int64_t a, int64_t b) { return a - b; });
  if (auto cast = value.getDefiningOp<arith::ExtSIOp>())
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
                        [](const APInt &value) { return value.isOne(); });
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return isAllTrueMask(broadcast.getSrc());
  if (auto splat = value.getDefiningOp<SplatOp>()) {
    if (auto constant = splat.getSrc().getDefiningOp<arith::ConstantIntOp>())
      return constant.value() == 1;
    return false;
  }
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

// evaluatePointerOffset is the key recognizer: it converts a pointer expression
// like a nested tt.addptr + broadcast chain into a flat tensor of byte offsets.
// Once we have that tensor, we can compare it against the known packed-weight
// formulas and decide whether the logical dense load can be replaced by a compact
// memory load.
//
// The important detail is that the offset is not always a simple local value.
// It may be built from multiple nested addptrs, where the parent adds a base
// offset and the child adds a per-element offset.  We fold those together into a
// single offset table before checking the layout.
static FailureOr<StaticTensor> evaluatePointerOffset(Value value) {
  if (auto addPtr = value.getDefiningOp<AddPtrOp>()) {
    // Step 1: evaluate the offset contributed by this addptr itself.
    auto ownOffset = evaluate(addPtr.getOffset());
    if (failed(ownOffset))
      return failure();

    // Step 2: walk through any broadcast wrappers to reach the underlying base
    // pointer, then check whether the parent pointer also contributes a
    // nontrivial offset.
    Value parentValue = addPtr.getPtr();
    while (auto broadcast = parentValue.getDefiningOp<BroadcastOp>())
      parentValue = broadcast.getSrc();
    auto parent = parentValue.getDefiningOp<AddPtrOp>();
    if (!parent)
      return ownOffset;

    auto parentOffset = evaluatePointerOffset(parent.getResult());
    if (failed(parentOffset))
      return failure();

    // If both parent and child offsets share the same logical shape, simply add
    // them elementwise.  This is the common case for a tensor pointer built from
    // a scalar base plus a tensor offset.
    if (parentOffset->shape == ownOffset->shape) {
      for (size_t i = 0; i < ownOffset->values.size(); ++i)
        ownOffset->values[i] += parentOffset->values[i];
    } else {
      // Some pointer chains are broadcasted or reshaped, so the parent and child
      // offsets may differ in shape while still describing the same logical
      // tensor.  In that case, we map the parent offset back into the child's
      // linearized index space using simple stride arithmetic.
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
  if (auto addPtr = value.getDefiningOp<AddPtrOp>())
    return scalarBase(addPtr.getPtr());
  if (auto splat = value.getDefiningOp<SplatOp>())
    return splat.getSrc();
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return scalarBase(broadcast.getSrc());
  return value;
}

// W3-QH uses a row-major packed layout where each 32-element column group is
// physically stored in a compact sub-tile.  The offset formula below matches the
// actual memory ordering used by the upstream qweight packer.
static bool isW3QH(ArrayRef<int64_t> offsets, int64_t rows, int64_t columns) {
  if (columns % 32 != 0)
    return false;
  int64_t groups = columns / 32;
  if (static_cast<int64_t>(offsets.size()) != rows * columns)
    return false;
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t col = 0; col < columns; ++col) {
      int64_t expected = (row * groups + (col >> 5)) * 4 + ((col & 31) >> 3);
      if (offsets[row * columns + col] != expected)
        return false;
    }
  return true;
}

static FailureOr<int64_t> isW3QS(ArrayRef<int64_t> offsets, int64_t rows,
                                 int64_t columns) {
  if (columns % 32 != 0 || offsets.empty())
    return failure();
  int64_t groups = columns / 32;
  int64_t pair = offsets[0] & 1;
  if (pair > 1 || static_cast<int64_t>(offsets.size()) != rows * columns)
    return failure();
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t col = 0; col < columns; ++col) {
      int64_t expected = (row * groups + (col >> 5)) * 8 +
                         ((col & 31) >> 3) * 2 + pair;
      if (offsets[row * columns + col] != expected)
        return failure();
    }
  return pair;
}

static bool isW4(ArrayRef<int64_t> offsets, int64_t rows, int64_t columns) {
  if (columns % 32 != 0 || static_cast<int64_t>(offsets.size()) != rows * columns)
    return false;
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t col = 0; col < columns; ++col) {
      int64_t expected = row * (columns / 2) + (col >> 5) * 16 + (col & 15);
      if (offsets[row * columns + col] != expected)
        return false;
    }
  return true;
}

// The compact load is the central optimization: instead of materializing a
// full logical tensor load through the packed layout, we emit a single scalar
// base-pointer load covering the compact physical buffer and then reconstruct
// the logical layout with reshape/broadcast operations.
static Value createCompactLoad(Location loc, Value base, int64_t elements,
                               PatternRewriter &rewriter) {
  auto basePtr = dyn_cast<PointerType>(base.getType());
  if (!basePtr)
    return nullptr;
  auto ptrType = RankedTensorType::get({elements}, basePtr);
  auto indexType = RankedTensorType::get({elements}, rewriter.getI32Type());
  auto range = rewriter.create<MakeRangeOp>(loc, indexType, 0, elements);
  auto splat = rewriter.create<SplatOp>(loc, ptrType, base);
  auto ptr = rewriter.create<AddPtrOp>(loc, ptrType, splat, range);
  return rewriter
      .create<LoadOp>(loc, ptr.getResult(), nullptr, nullptr,
                      CacheModifier::NONE, EvictionPolicy::NORMAL, false)
      .getResult();
}
} // namespace

// matchAndRewrite is the actual translation point.  We only trigger when a
// load is a 2-D tensor load whose pointer arithmetic follows the packed storage
// formula used by qweights.  Once recognized, the pass swaps the expensive
// irregular access pattern for a compact load plus shape restoration.
LogicalResult PackedLoadRewrite::matchAndRewrite(
    LoadOp op, PatternRewriter &rewriter) const {
  auto reject = [&](StringRef reason) -> LogicalResult {
    op.emitRemark() << "PackedLoadRewrite candidate: matched=no reason="
                    << reason;
    return failure();
  };
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto pointer = op.getPtr().getDefiningOp();
  op.emitRemark() << "PackedLoadRewrite candidate: shape="
                  << (resultType ? resultType.getShape() : ArrayRef<int64_t>())
                  << " pointer="
                  << (pointer ? pointer->getName().getStringRef()
                               : StringRef("<block-argument>"));
  if (op.getMask() && !isAllTrueMask(op.getMask()))
    return reject("unsupported mask");
  if (op.getOther() && !op.getMask())
    return reject("unsupported other value");
  auto addptr = op.getPtr().getDefiningOp<AddPtrOp>();
  if (!resultType)
    return reject("result is not a ranked tensor");
  if (!resultType.hasStaticShape())
    return reject("result shape is dynamic");
  if (resultType.getRank() != 2)
    return reject("result rank is not 2");
  if (!addptr)
    return reject("pointer producer is not tt.addptr");
  auto offsets = evaluatePointerOffset(op.getPtr());
  if (failed(offsets))
    return reject("offset expression is not statically evaluable");
  if (offsets->shape != resultType.getShape())
    return reject("offset shape differs from result shape");
  int64_t rows = resultType.getShape()[0];
  int64_t columns = resultType.getShape()[1];
  bool w3qh = isW3QH(offsets->values, rows, columns);
  auto w3qsPair = isW3QS(offsets->values, rows, columns);
  bool w4 = isW4(offsets->values, rows, columns);
  // If the offset table does not match any packaged-memory pattern, there is
  // no compact-load optimization to apply and the original load must be kept as
  //-is.
  if (!w3qh && failed(w3qsPair) && !w4)
    return reject("offset map is not W3-QH or W3-QS");
  bool w3qs = succeeded(w3qsPair);

  // The physical load count is the number of elements in the compact backing
  // buffer.  For W3-QH/W3-QS this is a packed 4/8-bit layout; for W4 it is a
  // denser row-wise representation.
  int64_t physical = (w3qh || w3qs) ? rows * (columns / 32) * (w3qs ? 8 : 4)
                          : rows * (columns / 2);

  // The rewrite emits a single compact load per base pointer and then reuses it
  // for every logical load that shares the same backing buffer and physical size.
  Value base = scalarBase(addptr.getPtr());
  Value compact = state ? state->compactLoads.lookup({base, physical}) : Value();
  if (!compact) {
    OpBuilder::InsertionGuard guard(rewriter);
    auto function = op->getParentOfType<triton::FuncOp>();
    if (!function || function.getBody().empty())
      return reject("load is not nested in a function body");
    rewriter.setInsertionPointToStart(&function.getBody().front());
    compact = createCompactLoad(op.getLoc(), base, physical, rewriter);
    if (state && compact)
      state->compactLoads[{base, physical}] = compact;
  }
  if (!compact)
    return reject("base pointer is not a scalar tt.ptr");

  // The compact buffer is a flat 1-D tensor.  We reshape it into the logical
  // packed shape, then expand the packed dimension back out so it matches the
  // original dense logical tensor layout.  The exact dimension placement depends
  // on whether we are handling W3 or W4.
  auto compactType = cast<RankedTensorType>(compact.getType());
  SmallVector<int64_t> packedShape =
      (w3qh || w3qs) ? SmallVector<int64_t>{rows, columns / 32, w3qs ? 4 : 4}
           : SmallVector<int64_t>{rows, columns / 32, 16};
  Value packedInput = compact;
  if (w3qs) {
    // For W3-QS, the packed layout carries a second axis selecting the pair
    // slot (0 or 1).  We first reshape to a 4-D packed tensor, extract the
    // relevant slice for this pair, and then continue with the standard reshape.
    packedShape.push_back(2);
    auto packedFull = rewriter.create<ReshapeOp>(
        op.getLoc(), RankedTensorType::get(packedShape, compactType.getElementType()),
        compact);
    SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
    offsets[3] = rewriter.getIndexAttr(*w3qsPair);
    SmallVector<OpFoldResult> sizes = {
        rewriter.getIndexAttr(rows), rewriter.getIndexAttr(columns / 32),
        rewriter.getIndexAttr(4), rewriter.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    auto sliceType = RankedTensorType::get(
        {rows, columns / 32, 4, 1}, compactType.getElementType());
    packedInput = rewriter.create<tensor::ExtractSliceOp>(
        op.getLoc(), sliceType, packedFull.getResult(), offsets, sizes,
        strides);
    packedShape.pop_back();
  }
  auto packed = rewriter.create<ReshapeOp>(
      op.getLoc(), RankedTensorType::get(packedShape, compactType.getElementType()),
      packedInput);
  int64_t broadcastAxis = (w3qh || w3qs) ? 3 : 2;
  auto expanded = rewriter.create<ExpandDimsOp>(
      op.getLoc(), packed.getResult(), broadcastAxis);
  SmallVector<int64_t> expandedShape = packedShape;
  expandedShape.insert(expandedShape.begin() + broadcastAxis, w3qh ? 1 : 1);
  expandedShape[broadcastAxis] = (w3qh || w3qs) ? 8 : 2;
  auto broadcast = rewriter.create<BroadcastOp>(
      op.getLoc(), RankedTensorType::get(expandedShape, compactType.getElementType()),
      expanded.getResult());
  auto restored = rewriter.create<ReshapeOp>(op.getLoc(), resultType,
                                             broadcast.getResult());
  rewriter.replaceOp(op, restored.getResult());

  op.emitRemark() << "PackedLoadRewrite: logical shape=" << rows << "x" << columns
                  << " physical elements=" << physical
                  << " reuse factor=" << (rows * columns) / physical
                  << " kind=" << (w3qh ? "W3 QH" : (w3qs ? "W3 QS" : "W4 qweight"));
  return success();
}
