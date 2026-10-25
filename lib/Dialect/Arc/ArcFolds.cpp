//===- ArcFolds.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isAlways(Attribute attr, bool expected) {
  if (auto enable = dyn_cast_or_null<IntegerAttr>(attr))
    return enable.getValue().getBoolValue() == expected;
  return false;
}

/// Construct a zero using the constant representation of the result type.
/// Unsupported types leave the state intact rather than creating an invalid
/// IntegerAttr with an aggregate (or other non-integer) type.
static Attribute getStateZeroAttr(Type type) {
  if (hw::getBitWidth(type) < 0)
    return {};
  if (auto integer = dyn_cast<IntegerType>(type))
    return IntegerAttr::get(integer, 0);
  auto getArrayZero = [&](auto array) -> Attribute {
    auto element = getStateZeroAttr(array.getElementType());
    if (!element)
      return {};
    return ArrayAttr::get(
        type.getContext(),
        SmallVector<Attribute>(array.getNumElements(), element));
  };
  if (auto array = dyn_cast<hw::ArrayType>(type))
    return getArrayZero(array);
  if (auto array = dyn_cast<hw::UnpackedArrayType>(type))
    return getArrayZero(array);
  if (auto structure = dyn_cast<hw::StructType>(type)) {
    SmallVector<Attribute> fields;
    for (auto field : structure.getElements()) {
      auto zero = getStateZeroAttr(field.type);
      if (!zero)
        return {};
      fields.push_back(zero);
    }
    return ArrayAttr::get(type.getContext(), fields);
  }
  return {};
}

static LogicalResult foldStateToZero(TypeRange types,
                                     SmallVectorImpl<OpFoldResult> &results) {
  SmallVector<Attribute> zeros;
  for (auto type : types) {
    auto zero = getStateZeroAttr(type);
    if (!zero)
      return failure();
    zeros.push_back(zero);
  }
  results.append(zeros.begin(), zeros.end());
  return success();
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {

  if (getNumResults() > 0 && !getOperation()->hasAttr("name") &&
      !getOperation()->hasAttr("names")) {
    bool hasExplicitInitials = !getInitials().empty();
    bool allInitialsConstant =
        !hasExplicitInitials ||
        llvm::all_of(adaptor.getInitials(),
                     [&](Attribute attr) { return !!attr; });
    if (isAlways(adaptor.getEnable(), false) && allInitialsConstant) {
      // Fold to the explicit or implicit initial value if
      // the state is never enabled and the initial values
      // are compile-time constants.
      if (hasExplicitInitials)
        results.append(adaptor.getInitials().begin(),
                       adaptor.getInitials().end());
      else
        return foldStateToZero(getResultTypes(), results);
      return success();
    }
    if (!hasExplicitInitials && isAlways(adaptor.getReset(), true)) {
      // We assume both the implicit initial value and the
      // implicit (synchronous) reset value to be zero.
      return foldStateToZero(getResultTypes(), results);
    }
  }

  // Remove operand when input is default value.
  if (isAlways(adaptor.getReset(), false))
    return getResetMutable().clear(), success();

  // Remove operand when input is default value.
  if (isAlways(adaptor.getEnable(), true))
    return getEnableMutable().clear(), success();

  return failure();
}

LogicalResult StateOp::canonicalize(StateOp op, PatternRewriter &rewriter) {
  // When there are no names attached, the state is not externaly observable.
  // When there are also no internal users, we can remove it.
  if (op->use_empty() && !op->hasAttr("name") && !op->hasAttr("names")) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// StorageGetOp
//===----------------------------------------------------------------------===//

LogicalResult StorageGetOp::canonicalize(StorageGetOp op,
                                         PatternRewriter &rewriter) {
  if (auto pred = op.getStorage().getDefiningOp<StorageGetOp>()) {
    rewriter.modifyOpInPlace(op, [&] {
      op.getStorageMutable().assign(pred.getStorage());
      op.setOffset(op.getOffset() + pred.getOffset());
    });
    return success();
  }
  return failure();
}
