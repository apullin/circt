// RUN: arcilator %s --run --jit-entry=run --jit-opt-level=0 | FileCheck %s
// RUN: arcilator %s --run --jit-entry=run --jit-opt-level=1 | FileCheck %s

// The loop creates more than 8 MiB of dynamic copies if an allocation is
// emitted on every iteration. Reusing a buffer must still preserve the old
// array while the new array is being constructed.
func.func private @lagged() -> (i64, i64) {
  %seed = hw.aggregate_constant [0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64, 0 : i64] : !hw.array<8xi64>
  %index = hw.constant 0 : i3
  %one = hw.constant 1 : i64
  %limit = hw.constant 200000 : i64
  cf.br ^loop(%seed, %one : !hw.array<8xi64>, i64)
^loop(%current: !hw.array<8xi64>, %iteration: i64):
  %next = hw.array_inject %current[%index], %iteration : !hw.array<8xi64>, i3
  %oldValue = hw.array_get %current[%index] : !hw.array<8xi64>, i3
  %newValue = hw.array_get %next[%index] : !hw.array<8xi64>, i3
  %more = comb.icmp ult %iteration, %limit : i64
  %nextIteration = comb.add %iteration, %one : i64
  cf.cond_br %more, ^loop(%next, %nextIteration : !hw.array<8xi64>, i64), ^done
^done:
  return %newValue, %oldValue : i64, i64
}

// Block-argument copies are parallel assignments. A swap must read both old
// buffers before either destination is overwritten.
func.func private @swap() -> (i64, i64) {
  %a = hw.aggregate_constant [11 : i64, 11 : i64] : !hw.array<2xi64>
  %b = hw.aggregate_constant [22 : i64, 22 : i64] : !hw.array<2xi64>
  %index = hw.constant 0 : i1
  %zero = hw.constant 0 : i64
  %one = hw.constant 1 : i64
  %three = hw.constant 3 : i64
  cf.br ^loop(%a, %b, %zero : !hw.array<2xi64>, !hw.array<2xi64>, i64)
^loop(%left: !hw.array<2xi64>, %right: !hw.array<2xi64>, %iteration: i64):
  %more = comb.icmp ult %iteration, %three : i64
  %next = comb.add %iteration, %one : i64
  cf.cond_br %more, ^loop(%right, %left, %next : !hw.array<2xi64>, !hw.array<2xi64>, i64), ^done
^done:
  %leftValue = hw.array_get %left[%index] : !hw.array<2xi64>, i1
  %rightValue = hw.array_get %right[%index] : !hw.array<2xi64>, i1
  return %leftValue, %rightValue : i64, i64
}

llvm.mlir.global private constant @message("array snapshots preserved\00")
llvm.func @puts(!llvm.ptr) -> i32
llvm.func @abort()

// CHECK: array snapshots preserved
func.func @run() {
  %current, %old = func.call @lagged() : () -> (i64, i64)
  %left, %right = func.call @swap() : () -> (i64, i64)
  %expectedCurrent = hw.constant 200000 : i64
  %expectedOld = hw.constant 199999 : i64
  %eleven = hw.constant 11 : i64
  %twentyTwo = hw.constant 22 : i64
  %checkCurrent = comb.icmp eq %current, %expectedCurrent : i64
  %checkOld = comb.icmp eq %old, %expectedOld : i64
  %checkLeft = comb.icmp eq %left, %twentyTwo : i64
  %checkRight = comb.icmp eq %right, %eleven : i64
  %correct = comb.and %checkCurrent, %checkOld, %checkLeft, %checkRight : i1
  cf.cond_br %correct, ^pass, ^fail
^pass:
  %message = llvm.mlir.addressof @message : !llvm.ptr
  %unused = llvm.call @puts(%message) : (!llvm.ptr) -> i32
  return
^fail:
  llvm.call @abort() : () -> ()
  return
}
