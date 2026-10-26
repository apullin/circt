// RUN: circt-opt %s --arc-lower-arrays | FileCheck %s
// RUN: arcilator %s --run --jit-entry=run --jit-opt-level=0 | FileCheck %s --check-prefix=EXEC
// RUN: arcilator %s --run --jit-entry=run --jit-opt-level=1 | FileCheck %s --check-prefix=EXEC

// Snapshot conditionally, but still before the write that replaces the old
// contents. The late predicate depends on an old scalar read and can be hoisted.
// CHECK-LABEL: func.func private @guarded
// CHECK: %[[RAW:.*]] = arc.state_read %arg0
// CHECK: %[[BUFFER:.*]] = arc.arrayref.alloc
// CHECK: %[[FLAG:.*]] = arc.state_read %arg1
// CHECK: %[[TAKE:.*]] = comb.and %[[FLAG]], %arg3
// CHECK: scf.if %[[TAKE]] {
// CHECK: arc.arrayref.copy %[[BUFFER]] = %[[RAW]]
// CHECK: }
// CHECK: arc.state_write %arg0
// CHECK: scf.if %[[TAKE]]
func.func private @guarded(%slot: !arc.state<!arc.arrayref<4xi32>>, %flag: !arc.state<i1>, %replacement: !arc.arrayref<4xi32>, %running: i1) -> i32 {
  %raw = arc.state_read %slot : <!arc.arrayref<4xi32>>
  %temporary = arc.arrayref.alloc : <4xi32>
  %snapshot = arc.arrayref.copy %temporary = %raw : <4xi32>
  %enabled = arc.state_read %flag : <i1>
  arc.state_write %slot = %replacement : <!arc.arrayref<4xi32>>
  %take = comb.and %enabled, %running : i1
  %zero = arith.constant 0 : index
  %noValue = hw.constant 0 : i32
  %answer = scf.if %take -> i32 {
    %old = func.call @read_one(%snapshot) : (!arc.arrayref<4xi32>) -> i32
    scf.yield %old : i32
  } else {
    scf.yield %noValue : i32
  }
  return %answer : i32
}

// A predicate read after a state write must not be hoisted across that write.
// CHECK-LABEL: func.func private @new_predicate
// CHECK: arc.arrayref.copy
// CHECK: arc.state_write %arg1
// CHECK: arc.state_read %arg1
// CHECK: scf.if
func.func private @new_predicate(%slot: !arc.state<!arc.arrayref<4xi32>>, %flag: !arc.state<i1>, %replacement: !arc.arrayref<4xi32>) -> i32 {
  %raw = arc.state_read %slot : <!arc.arrayref<4xi32>>
  %temporary = arc.arrayref.alloc : <4xi32>
  %snapshot = arc.arrayref.copy %temporary = %raw : <4xi32>
  %yes = hw.constant true
  arc.state_write %flag = %yes : <i1>
  %take = arc.state_read %flag : <i1>
  arc.state_write %slot = %replacement : <!arc.arrayref<4xi32>>
  %zero = arith.constant 0 : index
  %noValue = hw.constant 0 : i32
  %answer = scf.if %take -> i32 {
    %old = arc.arrayref.get %snapshot[%zero] : <4xi32> -> i32
    scf.yield %old : i32
  } else {
    scf.yield %noValue : i32
  }
  return %answer : i32
}

// An else-region reader needs the opposite predicate, so keep this snapshot.
// CHECK-LABEL: func.func private @else_reader
// CHECK: arc.arrayref.copy
// CHECK: arc.state_write
// CHECK: scf.if
func.func private @else_reader(%slot: !arc.state<!arc.arrayref<4xi32>>, %replacement: !arc.arrayref<4xi32>, %condition: i1) -> i32 {
  %raw = arc.state_read %slot : <!arc.arrayref<4xi32>>
  %temporary = arc.arrayref.alloc : <4xi32>
  %snapshot = arc.arrayref.copy %temporary = %raw : <4xi32>
  arc.state_write %slot = %replacement : <!arc.arrayref<4xi32>>
  %zero = arith.constant 0 : index
  %noValue = hw.constant 0 : i32
  %answer = scf.if %condition -> i32 {
    scf.yield %noValue : i32
  } else {
    %old = arc.arrayref.get %snapshot[%zero] : <4xi32> -> i32
    scf.yield %old : i32
  }
  return %answer : i32
}

func.func private @read_one(%array: !arc.arrayref<4xi32>) -> i32 {
  %zero = arith.constant 0 : index
  %value = arc.arrayref.get %array[%zero] : <4xi32> -> i32
  return %value : i32
}

func.func private @return_alias(%array: !arc.arrayref<4xi32>) -> !arc.arrayref<4xi32> {
  return %array : !arc.arrayref<4xi32>
}

// A call can return an alias that survives the guard. Even on the second
// iteration, where the guard is false, an indirect reader observes the copy.
// CHECK-LABEL: func.func private @escaping_alias
// CHECK: arc.arrayref.copy
// CHECK: scf.if
// CHECK: func.call @return_alias
func.func private @escaping_alias(%slot: !arc.state<!arc.arrayref<4xi32>>, %replacement: !arc.arrayref<4xi32>) -> i32 {
  %temporary = arc.arrayref.alloc : <4xi32>
  %seed = arc.arrayref.alloc init([0 : i32, 0 : i32, 0 : i32, 0 : i32]) : <4xi32>
  %zero = arith.constant 0 : index
  %firstIteration = hw.constant 0 : i32
  %secondIteration = hw.constant 1 : i32
  cf.br ^loop(%seed, %firstIteration : !arc.arrayref<4xi32>, i32)
^loop(%held: !arc.arrayref<4xi32>, %iteration: i32):
  %raw = arc.state_read %slot : <!arc.arrayref<4xi32>>
  %snapshot = arc.arrayref.copy %temporary = %raw : <4xi32>
  arc.state_write %slot = %replacement : <!arc.arrayref<4xi32>>
  %first = comb.icmp eq %iteration, %firstIteration : i32
  %nextHeld = scf.if %first -> !arc.arrayref<4xi32> {
    %alias = func.call @return_alias(%snapshot) : (!arc.arrayref<4xi32>) -> !arc.arrayref<4xi32>
    scf.yield %alias : !arc.arrayref<4xi32>
  } else {
    scf.yield %held : !arc.arrayref<4xi32>
  }
  %answer = arc.arrayref.get %nextHeld[%zero] : <4xi32> -> i32
  cf.cond_br %first, ^loop(%nextHeld, %secondIteration : !arc.arrayref<4xi32>, i32), ^done
^done:
  return %answer : i32
}

llvm.mlir.global private constant @message("guarded array snapshots preserved old values\00")
llvm.func @puts(!llvm.ptr) -> i32
llvm.func @abort()

// EXEC: guarded array snapshots preserved old values
func.func @run() {
  %size = hw.constant 32 : i64
  %memory = llvm.alloca %size x i8 : (i64) -> !llvm.ptr
  %storage = builtin.unrealized_conversion_cast %memory : !llvm.ptr to !arc.storage
  %slot = arc.storage.get %storage[0] : !arc.storage -> !arc.state<!arc.arrayref<4xi32>>
  %flag = arc.storage.get %storage[16] : !arc.storage -> !arc.state<i1>
  %source = arc.arrayref.alloc init([11 : i32, 11 : i32, 11 : i32, 11 : i32]) : <4xi32>
  %replacement = arc.arrayref.alloc init([29 : i32, 29 : i32, 29 : i32, 29 : i32]) : <4xi32>
  %yes = hw.constant true
  %no = hw.constant false
  arc.state_write %slot = %source : <!arc.arrayref<4xi32>>
  arc.state_write %flag = %yes : <i1>
  %old = func.call @guarded(%slot, %flag, %replacement, %yes) : (!arc.state<!arc.arrayref<4xi32>>, !arc.state<i1>, !arc.arrayref<4xi32>, i1) -> i32
  arc.state_write %slot = %source : <!arc.arrayref<4xi32>>
  arc.state_write %flag = %no : <i1>
  %unused = func.call @guarded(%slot, %flag, %replacement, %yes) : (!arc.state<!arc.arrayref<4xi32>>, !arc.state<i1>, !arc.arrayref<4xi32>, i1) -> i32
  arc.state_write %slot = %source : <!arc.arrayref<4xi32>>
  %newFlag = func.call @new_predicate(%slot, %flag, %replacement) : (!arc.state<!arc.arrayref<4xi32>>, !arc.state<i1>, !arc.arrayref<4xi32>) -> i32
  arc.state_write %slot = %source : <!arc.arrayref<4xi32>>
  %otherBranch = func.call @else_reader(%slot, %replacement, %no) : (!arc.state<!arc.arrayref<4xi32>>, !arc.arrayref<4xi32>, i1) -> i32
  arc.state_write %slot = %source : <!arc.arrayref<4xi32>>
  %escaped = func.call @escaping_alias(%slot, %replacement) : (!arc.state<!arc.arrayref<4xi32>>, !arc.arrayref<4xi32>) -> i32
  %twentyNine = hw.constant 29 : i32
  %eleven = hw.constant 11 : i32
  %zero = hw.constant 0 : i32
  %a = comb.icmp eq %old, %eleven : i32
  %b = comb.icmp eq %unused, %zero : i32
  %c = comb.icmp eq %newFlag, %eleven : i32
  %d = comb.icmp eq %otherBranch, %eleven : i32
  %e = comb.icmp eq %escaped, %twentyNine : i32
  %ok = comb.and %a, %b, %c, %d, %e : i1
  cf.cond_br %ok, ^pass, ^fail
^pass:
  %message = llvm.mlir.addressof @message : !llvm.ptr
  %printed = llvm.call @puts(%message) : (!llvm.ptr) -> i32
  return
^fail:
  llvm.call @abort() : () -> ()
  return
}
