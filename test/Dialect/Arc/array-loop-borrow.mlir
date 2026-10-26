// RUN: circt-opt %s --arc-lower-arrays | FileCheck %s
// RUN: arcilator %s --run --jit-entry=run --jit-opt-level=0 | FileCheck %s --check-prefix=EXEC
// RUN: arcilator %s --run --jit-entry=run --jit-opt-level=1 | FileCheck %s --check-prefix=EXEC

// Read the old element before replacing it. With only one array transfer in
// this loop, the clone-on-inject buffer can hold the next iteration's value.
// CHECK-LABEL: func.func private @read_before_update
// CHECK: ^bb1(%[[CURRENT:[a-zA-Z0-9_]+]]: !arc.arrayref<8xi64>,
// CHECK-NEXT: %{{.*}} = arc.arrayref.get %[[CURRENT]]
// CHECK: arc.arrayref.copy {{.*}} = %[[CURRENT]]
func.func private @read_before_update() -> i64 {
  %seed = hw.aggregate_constant [0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64] : !hw.array<8xi64>
  %index = hw.constant 0 : i3
  %one = hw.constant 1 : i64
  %limit = hw.constant 5000 : i64
  cf.br ^loop(%seed, %one : !hw.array<8xi64>, i64)
^loop(%current: !hw.array<8xi64>, %iteration: i64):
  %old = hw.array_get %current[%index] : !hw.array<8xi64>, i3
  %next = hw.array_inject %current[%index], %iteration : !hw.array<8xi64>, i3
  %more = comb.icmp ult %iteration, %limit : i64
  %nextIteration = comb.add %iteration, %one : i64
  cf.cond_br %more, ^loop(%next, %nextIteration : !hw.array<8xi64>, i64), ^done
^done:
  return %old : i64
}

// A slice may retain an old alias after the direct array use is dead. Keep
// staging even though scalar liveness alone would allow borrowing the array.
// CHECK-LABEL: func.func private @slice_after_update
// CHECK: ^bb1(
// CHECK: arc.arrayref.copy
// CHECK: arc.arrayref.copy
// CHECK: arc.arrayref.slice
func.func private @slice_after_update() -> i64 {
  %seed = hw.aggregate_constant [0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64] : !hw.array<8xi64>
  %index = hw.constant 0 : i3
  %sliceIndex = hw.constant 0 : i2
  %one = hw.constant 1 : i64
  %limit = hw.constant 5000 : i64
  cf.br ^loop(%seed, %one : !hw.array<8xi64>, i64)
^loop(%current: !hw.array<8xi64>, %iteration: i64):
  %view = hw.array_slice %current[%index] : (!hw.array<8xi64>) -> !hw.array<4xi64>
  %next = hw.array_inject %current[%index], %iteration : !hw.array<8xi64>, i3
  %old = hw.array_get %view[%sliceIndex] : !hw.array<4xi64>, i2
  %more = comb.icmp ult %iteration, %limit : i64
  %nextIteration = comb.add %iteration, %one : i64
  cf.cond_br %more, ^loop(%next, %nextIteration : !hw.array<8xi64>, i64), ^done
^done:
  return %old : i64
}

llvm.mlir.global private constant @message("loop borrowing preserves old arrays and views\00")
llvm.func @puts(!llvm.ptr) -> i32
llvm.func @abort()

// EXEC: loop borrowing preserves old arrays and views
func.func @run() {
  %a = func.call @read_before_update() : () -> i64
  %b = func.call @slice_after_update() : () -> i64
  %expected = hw.constant 4999 : i64
  %okA = comb.icmp eq %a, %expected : i64
  %okB = comb.icmp eq %b, %expected : i64
  %ok = comb.and %okA, %okB : i1
  cf.cond_br %ok, ^pass, ^fail
^pass:
  %message = llvm.mlir.addressof @message : !llvm.ptr
  %printed = llvm.call @puts(%message) : (!llvm.ptr) -> i32
  return
^fail:
  llvm.call @abort() : () -> ()
  return
}
