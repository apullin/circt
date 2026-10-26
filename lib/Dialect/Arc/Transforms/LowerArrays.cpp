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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
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

/// Do not snapshot a state array on activations where no consumer can read it.
/// Keep the copy at its original position, before any intervening state writes;
/// merely sinking it into the consumer would lose the required old value.
struct GuardStateArrayCopy : public OpRewritePattern<ArrayRefCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ArrayRefCopyOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getOutput().use_empty() ||
        !op.getSource().getDefiningOp<StateReadOp>())
      return failure();
    auto temporary = op.getInput().getDefiningOp<ArrayRefAllocOp>();
    if (!temporary || temporary.getInit() || !temporary.getOutput().hasOneUse())
      return failure();
    llvm::SmallDenseSet<Value> visiting;
    if (!isReadOnlyArray(op.getOutput(), visiting))
      return failure();

    scf::IfOp consumer;
    for (auto *user : op.getOutput().getUsers()) {
      auto *ancestor = user;
      while (ancestor && ancestor->getBlock() != op->getBlock())
        ancestor = ancestor->getParentOp();
      auto ifOp = dyn_cast_or_null<scf::IfOp>(ancestor);
      if (!ifOp || !op->isBeforeInBlock(ifOp) ||
          !ifOp.getThenRegion().isAncestor(user->getParentRegion()) ||
          (consumer && consumer != ifOp))
        return failure();
      consumer = ifOp;
    }

    DominanceInfo dominance(op->getParentOp());
    SmallVector<Operation *> hoisted;
    SmallPtrSet<Operation *, 8> visited;
    if (!makeAvailable(consumer.getCondition(), op, dominance, visited,
                       hoisted))
      return failure();

    for (auto *definition : hoisted)
      rewriter.moveOpBefore(definition, op);
    rewriter.setInsertionPoint(op);
    auto guard = scf::IfOp::create(rewriter, op.getLoc(),
                                   consumer.getCondition(), false);
    rewriter.replaceAllUsesWith(op.getOutput(), temporary.getOutput());
    rewriter.moveOpBefore(op, guard.thenYield());
    return success();
  }

  /// Reject escaping aliases: an indirect reader could observe this buffer on
  /// a later activation where its syntactic guard is false. Known by-value
  /// readers and calls with corresponding read-only arguments do not escape.
  static bool isReadOnlyArray(Value value,
                              llvm::SmallDenseSet<Value> &visiting) {
    if (!visiting.insert(value).second)
      return false;
    for (auto &use : value.getUses()) {
      auto *user = use.getOwner();
      if (isa<ArrayRefGetOp, ArrayRefToArrayOp>(user))
        continue;
      if (auto copy = dyn_cast<ArrayRefCopyOp>(user)) {
        if (use.getOperandNumber() == 1)
          continue;
      } else if (auto write = dyn_cast<StateWriteOp>(user)) {
        if (write.getValue() == value)
          continue;
      } else if (auto call = dyn_cast<func::CallOp>(user)) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        auto index = use.getOperandNumber();
        if (callee && !callee.isExternal() &&
            index < callee.getNumArguments() &&
            isReadOnlyArray(callee.getArgument(index), visiting))
          continue;
      }
      visiting.erase(value);
      return false;
    }
    visiting.erase(value);
    return true;
  }

  static bool makeAvailable(Value value, Operation *before,
                            DominanceInfo &dominance,
                            SmallPtrSetImpl<Operation *> &visited,
                            SmallVectorImpl<Operation *> &hoisted) {
    if (dominance.dominates(value, before))
      return true;
    auto *definition = value.getDefiningOp();
    if (!definition || definition->getBlock() != before->getBlock() ||
        !before->isBeforeInBlock(definition) || definition->getNumRegions())
      return false;
    if (!visited.insert(definition).second)
      return true;

    if (auto read = dyn_cast<StateReadOp>(definition)) {
      // A scalar old-state predicate can move past other reads, but never past
      // a write or an operation with unknown effects.
      if (!read.getType().isInteger(1))
        return false;
      for (auto *between = before->getNextNode(); between != definition;
           between = between->getNextNode())
        if (!isa<StateReadOp, ArrayRefAllocOp>(between) &&
            !isMemoryEffectFree(between))
          return false;
    } else if (!isMemoryEffectFree(definition) || !isSpeculatable(definition)) {
      return false;
    }
    for (auto operand : definition->getOperands())
      if (!makeAvailable(operand, before, dominance, visited, hoisted))
        return false;
    hoisted.push_back(definition);
    return true;
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

struct ConvertStateRead : public OpConversionPattern<StateReadOp> {
  using OpConversionPattern<StateReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StateReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = convertOpResultTypes(op, adaptor.getOperands(),
                                       *this->getTypeConverter(), rewriter);
    if (failed(result))
      return failure();

    // For correctness we need to copy the read state into a local array.
    Value resultValue = result.value()->getResult(0);
    rewriter.replaceOp(op, cloneArrayRef(resultValue, rewriter, op.getLoc()));
    return success();
  }
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

  patterns.add<ConvertFunc, ConvertReturn, ConvertCall,
               ConvertAggregateConstant, ConvertArrayGet, ConvertArrayInject,
               ConvertArraySlice, ConvertArrayConcat, ConvertArrayCreate,
               ConvertStorageGet, ConvertMux, ConvertAllocState,
               ConvertStateRead, ConvertStateWrite, ConvertRootInput,
               ConvertRootOutput, ConvertUnrealizedConversionCast>(
      converter, &getContext());
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
  cleanupPatterns.add<GuardStateArrayCopy, OptimizeReturnOfAlloc>(
      &getContext());
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(cleanupPatterns)))) {
    return signalPassFailure();
  }
}
