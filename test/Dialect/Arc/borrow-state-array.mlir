// RUN: circt-opt %s --arc-lower-arrays | FileCheck %s
// RUN: circt-opt %s --arc-lower-arrays | arcilator --run --jit-entry=run --jit-opt-level=0 | FileCheck %s --check-prefix=EXEC
// RUN: circt-opt %s --arc-lower-arrays | arcilator --run --jit-entry=run --jit-opt-level=1 | FileCheck %s --check-prefix=EXEC

// Exercise allocated-storage IR at the array-bufferization boundary before
// running the resulting reference-semantics IR through the full JIT pipeline.

// A conditional reader completes before the later write, without dominating
// it. The neighboring scalar write does not overlap the array's storage.
// CHECK-LABEL: func.func private @borrow
// CHECK: %[[RAW:.*]] = arc.state_read
// CHECK-NOT: arc.arrayref.copy
// CHECK: arc.state_write
// CHECK: scf.if
// CHECK: func.call @read_one(%[[RAW]])
// CHECK: arc.state_write
func.func private @borrow(%storage: !arc.storage, %replacement: !hw.array<4xi32>, %condition: i1) -> i32 {
  %slot = arc.storage.get %storage[0] : !arc.storage -> !arc.state<!hw.array<4xi32>>
  %neighbor = arc.storage.get %storage[16] : !arc.storage -> !arc.state<i32>
  %old = arc.state_read %slot : <!hw.array<4xi32>>
  %zero = hw.constant 0 : i32
  arc.state_write %neighbor = %zero : <i32>
  %value = scf.if %condition -> i32 {
    %read = func.call @read_one(%old) : (!hw.array<4xi32>) -> i32
    scf.yield %read : i32
  } else {
    scf.yield %zero : i32
  }
  arc.state_write %slot = %replacement : <!hw.array<4xi32>>
  return %value : i32
}

func.func private @read_one(%array: !hw.array<4xi32>) -> i32 {
  %zero = hw.constant 0 : i2
  %value = hw.array_get %array[%zero] : !hw.array<4xi32>, i2
  return %value : i32
}

// A differently typed slot still aliases the array. Compare byte ranges,
// including offsets through nested storage views, rather than SSA identities.
// CHECK-LABEL: func.func private @overlap
// CHECK: %[[OLD:.*]] = arc.arrayref.copy
// CHECK: arc.state_write
// CHECK: arc.arrayref.get %[[OLD]]
func.func private @overlap(%storage: !arc.storage) -> i32 {
  %view = arc.storage.get %storage[4] : !arc.storage -> !arc.storage
  %slot = arc.storage.get %view[4] : !arc.storage -> !arc.state<!hw.array<4xi32>>
  %alias = arc.storage.get %storage[8] : !arc.storage -> !arc.state<i32>
  %old = arc.state_read %slot : <!hw.array<4xi32>>
  %replacement = hw.constant 29 : i32
  arc.state_write %alias = %replacement : <i32>
  %zero = hw.constant 0 : i2
  %value = hw.array_get %old[%zero] : !hw.array<4xi32>, i2
  return %value : i32
}

func.func private @write_slot(%slot: !arc.state<!hw.array<4xi32>>, %replacement: !hw.array<4xi32>) {
  arc.state_write %slot = %replacement : <!hw.array<4xi32>>
  return
}

// Calls that can modify state are barriers, even when the array argument itself
// has value semantics. Passing its state slot separately can change the source.
// CHECK-LABEL: func.func private @callee_write
// CHECK: %[[OLD:.*]] = arc.arrayref.copy
// CHECK: call @write_slot
// CHECK: arc.arrayref.get %[[OLD]]
func.func private @callee_write(%storage: !arc.storage, %replacement: !hw.array<4xi32>) -> i32 {
  %slot = arc.storage.get %storage[0] : !arc.storage -> !arc.state<!hw.array<4xi32>>
  %old = arc.state_read %slot : <!hw.array<4xi32>>
  func.call @write_slot(%slot, %replacement) : (!arc.state<!hw.array<4xi32>>, !hw.array<4xi32>) -> ()
  %zero = hw.constant 0 : i2
  %value = hw.array_get %old[%zero] : !hw.array<4xi32>, i2
  return %value : i32
}

// A slice retains an alias beyond the slice operation. Conservatively keep the
// snapshot when a reader can forward a view rather than consuming the value.
// CHECK-LABEL: func.func private @slice_alias
// CHECK: %[[OLD:.*]] = arc.arrayref.copy
// CHECK: arc.arrayref.slice %[[OLD]]
// CHECK: arc.state_write
func.func private @slice_alias(%storage: !arc.storage, %replacement: !hw.array<4xi32>) -> i32 {
  %slot = arc.storage.get %storage[0] : !arc.storage -> !arc.state<!hw.array<4xi32>>
  %old = arc.state_read %slot : <!hw.array<4xi32>>
  %one = hw.constant 1 : i2
  %view = hw.array_slice %old[%one] : (!hw.array<4xi32>) -> !hw.array<2xi32>
  arc.state_write %slot = %replacement : <!hw.array<4xi32>>
  %zero = hw.constant 0 : i1
  %value = hw.array_get %view[%zero] : !hw.array<2xi32>, i1
  return %value : i32
}

llvm.mlir.global private constant @message("borrowed arrays preserve state snapshots\00")
llvm.func @puts(!llvm.ptr) -> i32
llvm.func @abort()

// EXEC: borrowed arrays preserve state snapshots
func.func @run() {
  %size = hw.constant 32 : i64
  %memory = llvm.alloca %size x i8 : (i64) -> !llvm.ptr
  %storage = builtin.unrealized_conversion_cast %memory : !llvm.ptr to !arc.storage
  %slot = arc.storage.get %storage[0] : !arc.storage -> !arc.state<!hw.array<4xi32>>
  %overlapping = arc.storage.get %storage[8] : !arc.storage -> !arc.state<!hw.array<4xi32>>
  %source = hw.aggregate_constant [11 : i32, 11 : i32, 11 : i32, 11 : i32] : !hw.array<4xi32>
  %replacement = hw.aggregate_constant [29 : i32, 29 : i32, 29 : i32, 29 : i32] : !hw.array<4xi32>
  %yes = hw.constant true
  %no = hw.constant false
  arc.state_write %slot = %source : <!hw.array<4xi32>>
  %a = func.call @borrow(%storage, %replacement, %yes) : (!arc.storage, !hw.array<4xi32>, i1) -> i32
  arc.state_write %slot = %source : <!hw.array<4xi32>>
  %b = func.call @borrow(%storage, %replacement, %no) : (!arc.storage, !hw.array<4xi32>, i1) -> i32
  arc.state_write %overlapping = %source : <!hw.array<4xi32>>
  %c = func.call @overlap(%storage) : (!arc.storage) -> i32
  arc.state_write %slot = %source : <!hw.array<4xi32>>
  %d = func.call @callee_write(%storage, %replacement) : (!arc.storage, !hw.array<4xi32>) -> i32
  arc.state_write %slot = %source : <!hw.array<4xi32>>
  %e = func.call @slice_alias(%storage, %replacement) : (!arc.storage, !hw.array<4xi32>) -> i32
  %eleven = hw.constant 11 : i32
  %zero = hw.constant 0 : i32
  %okA = comb.icmp eq %a, %eleven : i32
  %okB = comb.icmp eq %b, %zero : i32
  %okC = comb.icmp eq %c, %eleven : i32
  %okD = comb.icmp eq %d, %eleven : i32
  %okE = comb.icmp eq %e, %eleven : i32
  %ok = comb.and %okA, %okB, %okC, %okD, %okE : i1
  cf.cond_br %ok, ^pass, ^fail
^pass:
  %message = llvm.mlir.addressof @message : !llvm.ptr
  %printed = llvm.call @puts(%message) : (!llvm.ptr) -> i32
  return
^fail:
  llvm.call @abort() : () -> ()
  return
}
