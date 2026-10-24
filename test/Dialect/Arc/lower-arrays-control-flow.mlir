// RUN: circt-opt %s --arc-lower-arrays | FileCheck %s

// Locally created arrays carried by branches reproduce the original FP16/PDPU
// failure without requiring array-typed function signatures.
// CHECK-LABEL: func.func @branch_local
// CHECK: %[[ARRAY:.*]] = arc.arrayref.alloc init([17 : i8, 29 : i8]) : <2xi8>
// CHECK: cf.br ^[[JOIN:bb[0-9]+]]
// CHECK: ^[[JOIN]]:
// CHECK: arc.arrayref.get %[[ARRAY]]
func.func @branch_local(%index: i1) -> i8 {
  %array = hw.aggregate_constant [17 : i8, 29 : i8] : !hw.array<2xi8>
  cf.br ^join(%array : !hw.array<2xi8>)
^join(%joined: !hw.array<2xi8>):
  %value = hw.array_get %joined[%index] : !hw.array<2xi8>, i1
  return %value : i8
}

// CHECK-LABEL: func.func @conditional_local
// CHECK: cf.cond_br %{{.*}}, ^[[JOIN:bb[0-9]+]](%{{.*}} : !arc.arrayref<2xi8>), ^[[JOIN]](%{{.*}} : !arc.arrayref<2xi8>)
// CHECK: ^[[JOIN]](%[[ARRAY:.*]]: !arc.arrayref<2xi8>):
// CHECK: arc.arrayref.get %[[ARRAY]]
func.func @conditional_local(%condition: i1, %index: i1) -> i8 {
  %left = hw.aggregate_constant [1 : i8, 2 : i8] : !hw.array<2xi8>
  %right = hw.aggregate_constant [3 : i8, 4 : i8] : !hw.array<2xi8>
  cf.cond_br %condition, ^join(%left : !hw.array<2xi8>), ^join(%right : !hw.array<2xi8>)
^join(%joined: !hw.array<2xi8>):
  %value = hw.array_get %joined[%index] : !hw.array<2xi8>, i1
  return %value : i8
}

// CHECK-LABEL: func.func @switch_local
// CHECK: cf.switch %{{.*}} : i32, [
// CHECK: default: ^[[JOIN:bb[0-9]+]](%{{.*}} : !arc.arrayref<2xi8>),
// CHECK: 1: ^[[JOIN]](%{{.*}} : !arc.arrayref<2xi8>)
// CHECK: ^[[JOIN]](%[[ARRAY:.*]]: !arc.arrayref<2xi8>):
// CHECK: arc.arrayref.get %[[ARRAY]]
func.func @switch_local(%selector: i32, %index: i1) -> i8 {
  %left = hw.aggregate_constant [1 : i8, 2 : i8] : !hw.array<2xi8>
  %right = hw.aggregate_constant [3 : i8, 4 : i8] : !hw.array<2xi8>
  cf.switch %selector : i32, [
    default: ^join(%left : !hw.array<2xi8>),
    1: ^join(%right : !hw.array<2xi8>)
  ]
^join(%joined: !hw.array<2xi8>):
  %value = hw.array_get %joined[%index] : !hw.array<2xi8>, i1
  return %value : i8
}

// An array forwarded by the entry branch is not an array return value. In a
// multi-block function, inspect the function result types when creating sret
// arguments instead of inspecting the entry block's terminator operands.
// CHECK-LABEL: func.func @branch_argument
// CHECK-SAME: (%[[ARG:.*]]: !arc.arrayref<2xi8>, %[[INDEX:.*]]: i1) -> i8
// CHECK: cf.br ^[[JOIN:bb[0-9]+]]
// CHECK: ^[[JOIN]]:
// CHECK: arc.arrayref.get %[[ARG]]
func.func @branch_argument(%array: !hw.array<2xi8>, %index: i1) -> i8 {
  cf.br ^join(%array : !hw.array<2xi8>)
^join(%joined: !hw.array<2xi8>):
  %value = hw.array_get %joined[%index] : !hw.array<2xi8>, i1
  return %value : i8
}

// Cleanup coalesces the two return blocks. Either selected input is copied to
// the same explicit sret argument, preserving value semantics on either branch.
// CHECK-LABEL: func.func @conditional_result
// CHECK-SAME: (%[[SRET:.*]]: !arc.arrayref<2xi8>, %[[LEFT:.*]]: !arc.arrayref<2xi8>, %[[RIGHT:.*]]: !arc.arrayref<2xi8>, %[[COND:.*]]: i1) -> !arc.arrayref<2xi8>
// CHECK: cf.cond_br %[[COND]], ^[[JOIN:bb[0-9]+]](%[[LEFT]] : !arc.arrayref<2xi8>), ^[[JOIN]](%[[RIGHT]] : !arc.arrayref<2xi8>)
// CHECK: ^[[JOIN]](%[[SELECTED:.*]]: !arc.arrayref<2xi8>):
// CHECK: %[[COPY:.*]] = arc.arrayref.copy %[[SRET]] = %[[SELECTED]] : <2xi8>
// CHECK: return %[[COPY]] : !arc.arrayref<2xi8>
func.func @conditional_result(%left: !hw.array<2xi8>, %right: !hw.array<2xi8>, %condition: i1) -> !hw.array<2xi8> {
  cf.cond_br %condition, ^true, ^false
^true:
  return %left : !hw.array<2xi8>
^false:
  return %right : !hw.array<2xi8>
}

// CHECK-LABEL: func.func private @external
// CHECK-SAME: (!arc.arrayref<2xi8>, !arc.arrayref<2xi8>) -> !arc.arrayref<2xi8>
func.func private @external(!hw.array<2xi8>) -> !hw.array<2xi8>
