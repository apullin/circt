// RUN: circt-opt %s --arc-lower-arrays | FileCheck %s

// Coalescing a return buffer must preserve other users of the copy result.
// CHECK-LABEL: func.func @shared_return_copy
// CHECK-SAME: ([[OUT:%.+]]: !arc.arrayref<2xi8>)
// CHECK-NOT: arc.arrayref.alloc
// CHECK: [[FIRST:%.+]] = arc.arrayref.inject [[OUT]]
// CHECK: [[SECOND:%.+]] = arc.arrayref.inject [[FIRST]]
// CHECK-NOT: arc.arrayref.copy
// CHECK: [[VALUE:%.+]] = arc.arrayref.get [[SECOND]]
// CHECK: return [[SECOND]], [[VALUE]]
func.func @shared_return_copy(%out: !arc.arrayref<2xi8>) -> (!arc.arrayref<2xi8>, i8) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %five = hw.constant 5 : i8
  %seven = hw.constant 7 : i8
  %buffer = arc.arrayref.alloc : <2xi8>
  %first = arc.arrayref.inject %buffer[%zero], %five : <2xi8>, i8 -> <2xi8>
  %second = arc.arrayref.inject %first[%one], %seven : <2xi8>, i8 -> <2xi8>
  %copy = arc.arrayref.copy %out = %second : <2xi8>
  %value = arc.arrayref.get %copy[%zero] : <2xi8> -> i8
  return %copy, %value : !arc.arrayref<2xi8>, i8
}
