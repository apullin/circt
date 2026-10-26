//===- LowerArrays.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <utility>

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SCCIterator.h"

#define DEBUG_TYPE "arc-lower-arrays"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERARRAYS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;
using ::llvm::enumerate;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerArraysPass : public arc::impl::LowerArraysBase<LowerArraysPass> {
  LowerArraysPass() = default;
  LowerArraysPass(const LowerArraysPass &pass) : LowerArraysPass() {}

  void runOnOperation() override;
};

Value asIndex(Value value, OpBuilder &builder) {
  Location loc = builder.getUnknownLoc();
  if (Operation *parent = value.getDefiningOp()) {
    loc = parent->getLoc();
  }
  return arith::IndexCastUIOp::create(builder, loc, builder.getIndexType(),
                                      value);
}

// Allocations introduced while lowering array values have one storage slot per
// SSA definition. Keep these slots in the function entry block so a definition
// executed in a loop does not consume additional stack on every iteration.
// Cyclic block arguments are materialized into their own storage below to keep
// old dynamic values alive when a definition is executed again.
ArrayRefAllocOp createArrayRefAlloc(OpBuilder &builder, Location loc, Type type,
                                    ArrayAttr init) {
  OpBuilder::InsertionGuard guard(builder);
  auto *parent = builder.getBlock()->getParentOp();
  auto func = dyn_cast<func::FuncOp>(parent);
  if (!func)
    func = parent->getParentOfType<func::FuncOp>();
  if (func && parent == func.getOperation() &&
      builder.getBlock() != &func.getBody().front())
    builder.setInsertionPointToStart(&func.getBody().front());
  return ArrayRefAllocOp::create(builder, loc, type, init);
}

Value cloneArrayRef(Value value, OpBuilder &builder, Location loc) {
  Value newAlloc = createArrayRefAlloc(builder, loc, value.getType(), {});
  return ArrayRefCopyOp::create(builder, loc, newAlloc, value);
}

struct ConvertFunc : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Record which original block arguments have value semantics. Existing
    // array references must retain their original aliasing behavior.
    SmallVector<SmallVector<unsigned>> valueArrayArguments;
    for (auto &block : op.getBody()) {
      auto &arguments = valueArrayArguments.emplace_back();
      for (auto argument : block.getArguments())
        if (isa<ArrayType>(argument.getType()))
          arguments.push_back(argument.getArgNumber());
    }

    const TypeConverter &converter = *getTypeConverter();
    TypeConverter::SignatureConversion conversion(op.getNumArguments());

    SmallVector<Type> newArgTypes;
    SmallVector<Type> newResultTypes;
    // Any array-typed results become parameters. Inspect the function type
    // rather than the entry terminator: functions may have multiple blocks
    // or no body, and branch operands are not function results.
    for (Type resultType : op.getResultTypes()) {
      if (isa<ArrayType>(resultType)) {
        Type newType = converter.convertType(resultType);
        conversion.addInputs(newType);
        newArgTypes.push_back(newType);
      }
    }

    if (failed(converter.convertTypes(op.getArgumentTypes(), newArgTypes)) ||
        failed(converter.convertTypes(op.getResultTypes(), newResultTypes))) {
      return failure();
    }

    if (failed(converter.convertSignatureArgs(op.getArgumentTypes(),
                                              conversion)) ||
        failed(rewriter.convertRegionTypes(&op.getBody(), converter,
                                           &conversion))) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&] {
      op.setType(FunctionType::get(getContext(), newArgTypes, newResultTypes));
    });

    if (op.isExternal())
      return success();
    SmallPtrSet<Block *, 8> cyclicBlocks;
    for (auto scc = llvm::scc_begin(&op.getBody()); !scc.isAtEnd(); ++scc)
      if (scc.hasCycle())
        cyclicBlocks.insert(scc->begin(), scc->end());

    assert(valueArrayArguments.size() == op.getBody().getBlocks().size());
    for (auto [index, block] : llvm::enumerate(op.getBody())) {
      if (!cyclicBlocks.contains(&block))
        continue;
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      SmallVector<std::pair<Value, Value>> staged;
      for (unsigned argIndex : valueArrayArguments[index]) {
        auto argument = block.getArgument(argIndex);
        auto type = cast<ArrayRefType>(argument.getType());
        auto temporary =
            createArrayRefAlloc(rewriter, argument.getLoc(), type, {});
        auto snapshot =
            createArrayRefAlloc(rewriter, argument.getLoc(), type, {});
        auto copy = ArrayRefCopyOp::create(rewriter, argument.getLoc(),
                                           temporary, argument);
        staged.emplace_back(copy, snapshot);
      }
      // A branch transfers block arguments simultaneously. Stage every incoming
      // value before overwriting any owned snapshot, including for swaps.
      for (auto [stagedValue, snapshot] : staged) {
        auto copy = cast<ArrayRefCopyOp>(stagedValue.getDefiningOp());
        auto argument = copy.getSource();
        auto value = ArrayRefCopyOp::create(rewriter, argument.getLoc(),
                                            snapshot, stagedValue);
        rewriter.replaceAllUsesExcept(argument, value, copy);
      }
    }

    return success();
  }
};

struct ConvertReturn : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    int sretIndex = 0;
    SmallVector<Value> newOperands;
    for (Value operand : adaptor.getOperands()) {
      if (isa<ArrayRefType>(operand.getType())) {
        Value arg = func.getArgument(sretIndex++);
        Value copy =
            ArrayRefCopyOp::create(rewriter, op.getLoc(), arg, operand);
        newOperands.push_back(copy);
      } else {
        newOperands.push_back(operand);
      }
    }
    rewriter.modifyOpInPlace(op, [&] { op->setOperands(newOperands); });
    return success();
  }
};

struct ConvertCall : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> newOperands;
    for (Type resultType : op.getResultTypes()) {
      auto arrayType = dyn_cast<ArrayType>(resultType);
      if (!arrayType)
        continue;
      auto arrayRefType = ArrayRefType::get(arrayType.getElementType(),
                                            arrayType.getNumElements());
      Value alloc =
          createArrayRefAlloc(rewriter, op.getLoc(), arrayRefType, {});
      newOperands.push_back(alloc);
    }
    for (Value operand : adaptor.getOperands()) {
      newOperands.push_back(operand);
    }

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return failure();
    }

    auto newCall = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                        resultTypes, newOperands);
    newCall->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    rewriter.replaceOp(op, newCall);
    return success();
  }
};

struct ConvertAggregateConstant
    : public OpConversionPattern<AggregateConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newType =
        cast<ArrayRefType>(getTypeConverter()->convertType(op.getType()));
    Value newOp =
        createArrayRefAlloc(rewriter, op.getLoc(), newType, op.getFieldsAttr());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertArrayGet : public OpConversionPattern<ArrayGetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    Value index = asIndex(op.getIndex(), rewriter);
    Value newOp = ArrayRefGetOp::create(rewriter, op.getLoc(), resultType,
                                        adaptor.getInput(), index);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertArrayInject : public OpConversionPattern<ArrayInjectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value index = asIndex(op.getIndex(), rewriter);
    Value dest = cloneArrayRef(adaptor.getInput(), rewriter, op.getLoc());
    Value newOp =
        ArrayRefInjectOp::create(rewriter, op.getLoc(), dest.getType(), dest,
                                 index, adaptor.getElement());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertArraySlice : public OpConversionPattern<ArraySliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArraySliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Because we're converting from value semantics we can assume that the
    // input buffer is immutable, so we don't need to copy it.
    Value index = asIndex(op.getLowIndex(), rewriter);
    Type destType = getTypeConverter()->convertType(op.getType());
    Value newOp = ArrayRefSliceOp::create(rewriter, op.getLoc(), destType,
                                          adaptor.getInput(), index);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertArrayConcat : public OpConversionPattern<ArrayConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type destType = getTypeConverter()->convertType(op.getType());
    Value dest = createArrayRefAlloc(rewriter, op.getLoc(), destType, {});

    // ArrayConcatOp's operands are ordered from most significant to least
    // significant.
    int offset = cast<ArrayRefType>(destType).getNumElements();
    for (Value operand : adaptor.getOperands()) {
      offset -= cast<ArrayRefType>(operand.getType()).getNumElements();
      Value index =
          arith::ConstantIndexOp::create(rewriter, op.getLoc(), offset);
      Value destSlice = ArrayRefSliceOp::create(rewriter, op.getLoc(),
                                                operand.getType(), dest, index);
      ArrayRefCopyOp::create(rewriter, op.getLoc(), destSlice, operand);
    }
    assert(offset == 0);
    rewriter.replaceOp(op, dest);
    return success();
  }
};

struct ConvertArrayCreate : public OpConversionPattern<ArrayCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    Value alloc = createArrayRefAlloc(rewriter, op.getLoc(), newType, {});
    Value create = ArrayRefCreateOp::create(rewriter, op.getLoc(), newType,
                                            alloc, adaptor.getInputs());
    rewriter.replaceOp(op, create);
    return success();
  }
};

struct ConvertStorageGet : public OpConversionPattern<StorageGetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StorageGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = convertOpResultTypes(op, adaptor.getOperands(),
                                       *getTypeConverter(), rewriter);
    if (failed(result))
      return failure();

    rewriter.replaceOp(op, *result);
    return success();
  }
};

struct ConvertMux : public OpConversionPattern<comb::MuxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    Value newOp = arith::SelectOp::create(
        rewriter, op.getLoc(), newType, adaptor.getCond(),
        adaptor.getTrueValue(), adaptor.getFalseValue());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// Identifies a return of an ArrayRef that is defined by an ArrayRefAllocOp,
// and replaces the alloc with the sret buffer. Also removes the copy.
struct OptimizeReturnOfAlloc : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    bool changed = false;

    // Iterate over all pairs of (result, sret-buffer). The sret buffers are
    // always the initial function arguments, and every !arc.arrayref<T>
    // result has one sret buffer associated with it.
    auto args = funcOp.getArguments();
    auto results =
        llvm::make_filter_range(op->getOpOperands(), [](OpOperand &operand) {
          return isa<ArrayRefType>(operand.get().getType());
        });

    for (auto [arg, result] : llvm::zip(args, results)) {
      Value resultValue = result.get();
      auto copy = resultValue.getDefiningOp<ArrayRefCopyOp>();
      if (!copy || copy.getInput() != arg)
        continue;
      ArrayRefAllocOp alloc = getUltimatelyDefiningAlloc(copy.getSource());
      if (!alloc || alloc.getInit())
        continue;
      // Note that `result` is a structured binding, which cannot be implicitly
      // captured until C++20.
      rewriter.modifyOpInPlace(
          op, [&, &result = result] { result.set(copy.getSource()); });
      rewriter.replaceAllUsesWith(alloc, arg);
      rewriter.eraseOp(alloc);
      rewriter.replaceOp(copy, copy.getSource());
      changed = true;
    }
    return success(changed);
  }

  ArrayRefAllocOp getUltimatelyDefiningAlloc(Value value) const {
    if (!isa<ArrayRefType>(value.getType()))
      return nullptr;
    while (value) {
      Operation *op = value.getDefiningOp();
      if (!op)
        return nullptr;

      if (auto alloc = dyn_cast<ArrayRefAllocOp>(op))
        return alloc;

      value =
          TypeSwitch<Operation *, Value>(op)
              .Case<ArrayRefCopyOp>([&](auto copy) { return copy.getInput(); })
              .Case<ArrayRefInjectOp>(
                  [&](auto inject) { return inject.getInput(); })
              .Case<ArrayRefFromArrayOp>(
                  [&](auto fromArray) { return fromArray.getInput(); })
              .Case<func::CallOp>([&](auto call) {
                OpResult result = cast<OpResult>(value);
                return call.getOperand(result.getResultNumber());
              })
              .Default([&](Operation *) { return nullptr; });
    }
    return nullptr;
  }
};

/// Find state-array reads whose consumers finish before the underlying storage
/// can change. Analyze the original value-semantics IR: array updates and call
/// results get their own buffers during conversion, whereas array slices and
/// reference-typed values can retain aliases and are deliberately rejected.
struct BorrowStateArrays {
  struct Range {
    Value storage;
    uint64_t offset;
    uint64_t size;
  };

  static std::optional<Range> getRange(Value state) {
    auto get = state.getDefiningOp<StorageGetOp>();
    if (!get)
      return {};
    Range range{get.getStorage(), get.getOffset(),
                cast<StateType>(state.getType()).getByteWidth()};
    while (auto parent = range.storage.getDefiningOp<StorageGetOp>()) {
      range.offset += parent.getOffset();
      range.storage = parent.getStorage();
    }
    return range;
  }

  // This is lexical ordering, including conditional readers: a read in an
  // earlier if finishes before a later sibling write, even though it does not
  // dominate that write. Callers exclude loop-carried and escaping aliases.
  static bool isBefore(Operation *a, Operation *b) {
    for (auto *left = a; left; left = left->getParentOp())
      for (auto *right = b; right; right = right->getParentOp())
        if (left->getBlock() == right->getBlock())
          return left != right && left->isBeforeInBlock(right);
    return false;
  }

  bool canBorrow(StateReadOp read) {
    auto array = dyn_cast<ArrayType>(read.getType());
    if (!array || !isa<IntegerType>(array.getElementType()) ||
        read.getValue().use_empty())
      return false;
    auto *block = read->getBlock();
    if (!isa<ModelOp, func::FuncOp>(block->getParentOp()) ||
        !block->getParent()->hasOneBlock())
      return false;
    auto range = getRange(read.getState());
    if (!range)
      return false;

    SmallVector<Operation *> readers;
    for (auto &use : read.getValue().getUses()) {
      auto *user = use.getOwner();
      // Each permitted consumer finishes reading this value at the operation
      // itself. In particular, converted calls copy array results into sret
      // buffers and array_inject first clones its input.
      if (auto get = dyn_cast<ArrayGetOp>(user)) {
        if (isa<ArrayType>(get.getType()))
          return false;
      } else if (auto write = dyn_cast<StateWriteOp>(user)) {
        if (write.getValue() != read.getValue())
          return false;
      } else if (!isa<func::CallOp, ArrayInjectOp>(user)) {
        return false;
      }
      auto *ancestor = user;
      while (ancestor && ancestor->getBlock() != block) {
        ancestor = ancestor->getParentOp();
        if (!isa_and_nonnull<scf::IfOp>(ancestor))
          return false;
      }
      if (!ancestor || !read->isBeforeInBlock(ancestor))
        return false;
      readers.push_back(user);
    }

    for (auto &top : *block) {
      auto result = top.walk([&](Operation *op) {
        if (isBefore(op, read) || op == read ||
            llvm::all_of(readers,
                         [&](auto *user) { return isBefore(user, op); }))
          return WalkResult::advance();
        return cannotWriteState(op, range) ? WalkResult::advance()
                                           : WalkResult::interrupt();
      });
      if (result.wasInterrupted())
        return false;
    }
    return true;
  }

  bool cannotWriteState(Operation *op, std::optional<Range> range) {
    if (auto write = dyn_cast<StateWriteOp>(op)) {
      auto written = getRange(write.getState());
      return range && written && range->storage == written->storage &&
             (range->offset + range->size <= written->offset ||
              written->offset + written->size <= range->offset);
    }
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isExternal())
        return false;
      auto [it, inserted] = stateFreeCallees.try_emplace(callee, false);
      if (!inserted)
        return it->second;
      // Unknown/recursive calls and explicit mutable array writes are barriers.
      // Pure hw.array operations cannot mutate their value-typed arguments.
      auto result = callee.getBody().walk([&](Operation *nested) {
        return cannotWriteState(nested, {}) ? WalkResult::advance()
                                            : WalkResult::interrupt();
      });
      bool safe = !result.wasInterrupted();
      stateFreeCallees[callee] = safe;
      return safe;
    }
    // These operations affect fresh allocations, output streams, or runtime
    // context fields, never the model's hardware state slots.
    if (isa<ArrayRefAllocOp, sim::PrintFormattedProcOp, arc::TerminateOp,
            SetNextWakeupOp>(op))
      return true;
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return true; // The walk checks the nested effects separately.
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      return llvm::none_of(instances, [](auto &effect) {
        return isa<MemoryEffects::Write, MemoryEffects::Free>(
            effect.getEffect());
      });
    }
    return false;
  }

  DenseMap<Operation *, bool> stateFreeCallees;
};

struct ConvertStateRead : public OpConversionPattern<StateReadOp> {
  ConvertStateRead(TypeConverter &converter, MLIRContext *context,
                   const llvm::SmallDenseSet<Operation *> &borrowable)
      : OpConversionPattern(converter, context), borrowable(borrowable) {}

  LogicalResult
  matchAndRewrite(StateReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = convertOpResultTypes(op, adaptor.getOperands(),
                                       *this->getTypeConverter(), rewriter);
    if (failed(result))
      return failure();

    // Borrow only reads proven safe in the original value-semantics IR.
    Value resultValue = result.value()->getResult(0);
    rewriter.replaceOp(op,
                       borrowable.contains(op)
                           ? resultValue
                           : cloneArrayRef(resultValue, rewriter, op.getLoc()));
    return success();
  }

  const llvm::SmallDenseSet<Operation *> &borrowable;
};

template <typename Op>
struct ConvertTrivially : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = convertOpResultTypes(op, adaptor.getOperands(),
                                       *this->getTypeConverter(), rewriter);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }
};

using ConvertAllocState = ConvertTrivially<arc::AllocStateOp>;
using ConvertStateWrite = ConvertTrivially<arc::StateWriteOp>;
using ConvertRootInput = ConvertTrivially<arc::RootInputOp>;
using ConvertRootOutput = ConvertTrivially<arc::RootOutputOp>;
using ConvertUnrealizedConversionCast =
    ConvertTrivially<UnrealizedConversionCastOp>;

} // namespace

void LowerArraysPass::runOnOperation() {
  TypeConverter converter;
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  BorrowStateArrays borrowAnalysis;
  llvm::SmallDenseSet<Operation *> borrowable;
  getOperation().walk([&](StateReadOp read) {
    if (borrowAnalysis.canBorrow(read))
      borrowable.insert(read);
  });

  converter.addConversion([](Type type) { return type; });
  converter.addConversion([&converter](ArrayType type) -> Type {
    Type newElem = converter.convertType(type.getElementType());
    // Don't convert nested array types.
    if (newElem != type.getElementType())
      return type;
    return ArrayRefType::get(newElem, type.getNumElements());
  });
  converter.addConversion([&converter](StateType type) {
    return StateType::get(converter.convertType(type.getType()));
  });

  target.addLegalOp<ArrayRefFromArrayOp, ArrayRefToArrayOp>();
  // Arrays within structs or unions always use !hw.array.
  target.addLegalOp<StructCreateOp, StructInjectOp, StructExtractOp,
                    StructExplodeOp, UnionCreateOp, UnionExtractOp>();
  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
    FunctionType fty = func.getFunctionType();
    return converter.isLegal(fty.getInputs()) &&
           converter.isLegal(fty.getResults()) &&
           converter.isLegal(&func.getBody());
  });
  // An ArrayGetOp may legally return an array if the input type was a nested
  // array. Similarly for ArrayCreateOp and ArrayInjectOp.
  target.addDynamicallyLegalOp<ArrayGetOp>([&](ArrayGetOp op) {
    return converter.isLegal(op.getInput().getType());
  });
  target.addDynamicallyLegalOp<ArrayCreateOp, ArrayInjectOp>(
      [&](Operation *op) {
        return converter.isLegal(op->getResult(0).getType());
      });

  // Produces an ArrayRefType from an ArrayType.
  converter.addTargetMaterialization([&](OpBuilder &b, ArrayRefType type,
                                         ValueRange values, Location loc,
                                         Type fromType) -> Value {
    assert(isa<ArrayType>(fromType));
    Value alloc = createArrayRefAlloc(b, loc, type, {});
    return ArrayRefFromArrayOp::create(b, loc, type, alloc, values.front());
  });

  // Produces an ArrayType from an ArrayRefType.
  converter.addSourceMaterialization([&](OpBuilder &b, ArrayType type,
                                         ValueRange values,
                                         Location loc) -> Value {
    assert(isa<ArrayRefType>(values.front().getType()));
    return ArrayRefToArrayOp::create(b, loc, type, values.front());
  });

  patterns
      .add<ConvertFunc, ConvertReturn, ConvertCall, ConvertAggregateConstant,
           ConvertArrayGet, ConvertArrayInject, ConvertArraySlice,
           ConvertArrayConcat, ConvertArrayCreate, ConvertStorageGet,
           ConvertMux, ConvertAllocState, ConvertStateWrite, ConvertRootInput,
           ConvertRootOutput, ConvertUnrealizedConversionCast>(converter,
                                                               &getContext());
  patterns.add<ConvertStateRead>(converter, &getContext(), borrowable);
  cf::populateCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                     target);

  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns),
                                    config))) {
    return signalPassFailure();
  }

  // Apply some cleanup patterns to optimize away ArrayRefAllocOps.
  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<OptimizeReturnOfAlloc>(&getContext());
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(cleanupPatterns)))) {
    return signalPassFailure();
  }
}
