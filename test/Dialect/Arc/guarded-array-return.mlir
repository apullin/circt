// RUN: circt-opt %s --arc-lower-arrays | FileCheck %s
// RUN: circt-opt %s --arc-lower-arrays | arcilator --run --jit-entry=run --jit-opt-level=0 | FileCheck %s --check-prefix=EXEC
// RUN: circt-opt %s --arc-lower-arrays | arcilator --run --jit-entry=run --jit-opt-level=1 | FileCheck %s --check-prefix=EXEC

// Every caller reads this payload only under its returned enable. Return the
// private sret address unconditionally, but fill its bytes only when enabled.
// CHECK-LABEL: func.func private @produce(
// CHECK-SAME: %[[BUFFER:[a-zA-Z0-9_]+]]: !arc.arrayref<4xi32>,
// CHECK-SAME: %[[SOURCE:[a-zA-Z0-9_]+]]: !arc.arrayref<4xi32>, %[[ENABLE:[a-zA-Z0-9_]+]]: i1
// CHECK: scf.if %[[ENABLE]] {
// CHECK: arc.arrayref.copy %[[BUFFER]] = %[[SOURCE]]
// CHECK: }
// CHECK: return %[[BUFFER]], %[[ENABLE]]
func.func private @produce(%input: !hw.array<4xi32>, %enabled: i1) -> (!hw.array<4xi32>, i1) {
  return %input, %enabled : !hw.array<4xi32>, i1
}

func.func @consume(%input: !hw.array<4xi32>, %enabled: i1) -> i32 {
  %payload, %valid = func.call @produce(%input, %enabled) : (!hw.array<4xi32>, i1) -> (!hw.array<4xi32>, i1)
  %index = hw.constant 0 : i2
  %zero = hw.constant 0 : i32
  %value = scf.if %valid -> i32 {
    %read = hw.array_get %payload[%index] : !hw.array<4xi32>, i2
    scf.yield %read : i32
  } else {
    scf.yield %zero : i32
  }
  return %value : i32
}

// One unguarded caller is enough to require this result on disabled paths.
// CHECK-LABEL: func.func private @must_fill(
// CHECK-NOT: scf.if
// CHECK: arc.arrayref.copy
// CHECK: return
func.func private @must_fill(%input: !hw.array<4xi32>, %enabled: i1) -> (!hw.array<4xi32>, i1) {
  return %input, %enabled : !hw.array<4xi32>, i1
}

func.func @unconditional(%input: !hw.array<4xi32>, %enabled: i1) -> i32 {
  %payload, %valid = func.call @must_fill(%input, %enabled) : (!hw.array<4xi32>, i1) -> (!hw.array<4xi32>, i1)
  %index = hw.constant 0 : i2
  %value = hw.array_get %payload[%index] : !hw.array<4xi32>, i2
  return %value : i32
}

// A reference-typed API may expose its destination through another alias,
// even when the returned reference is read only in a guarded region.
// CHECK-LABEL: func.func private @shared_destination(
// CHECK-NOT: scf.if
// CHECK: arc.arrayref.copy
// CHECK: return
func.func private @shared_destination(%destination: !arc.arrayref<4xi32>, %input: !arc.arrayref<4xi32>, %enabled: i1) -> (!arc.arrayref<4xi32>, i1) {
  %written = arc.arrayref.copy %destination = %input : <4xi32>
  return %written, %enabled : !arc.arrayref<4xi32>, i1
}

func.func @observe_shared(%destination: !arc.arrayref<4xi32>, %input: !arc.arrayref<4xi32>, %enabled: i1) -> i32 {
  %payload, %valid = func.call @shared_destination(%destination, %input, %enabled) : (!arc.arrayref<4xi32>, !arc.arrayref<4xi32>, i1) -> (!arc.arrayref<4xi32>, i1)
  %index = arith.constant 0 : index
  scf.if %valid {
    %read = arc.arrayref.get %payload[%index] : <4xi32> -> i32
    // A side effect retains the guarded read without exposing its reference.
    %text = sim.fmt.dec %read specifierWidth 0 : i32
    sim.proc.print %text
  }
  %value = arc.arrayref.get %destination[%index] : <4xi32> -> i32
  return %value : i32
}

llvm.mlir.global private constant @message("guarded array returns preserve observed values\00")
llvm.func @puts(!llvm.ptr) -> i32
llvm.func @abort()

// EXEC: guarded array returns preserve observed values
func.func @run() {
  %input = hw.aggregate_constant [11 : i32, 11 : i32, 11 : i32, 11 : i32] : !hw.array<4xi32>
  %source = arc.arrayref.alloc init([11 : i32, 11 : i32, 11 : i32, 11 : i32]) : <4xi32>
  %destination = arc.arrayref.alloc init([29 : i32, 29 : i32, 29 : i32, 29 : i32]) : <4xi32>
  %yes = hw.constant true
  %no = hw.constant false
  %a = func.call @consume(%input, %yes) : (!hw.array<4xi32>, i1) -> i32
  %b = func.call @consume(%input, %no) : (!hw.array<4xi32>, i1) -> i32
  %c = func.call @unconditional(%input, %no) : (!hw.array<4xi32>, i1) -> i32
  %d = func.call @observe_shared(%destination, %source, %no) : (!arc.arrayref<4xi32>, !arc.arrayref<4xi32>, i1) -> i32
  %eleven = hw.constant 11 : i32
  %zero = hw.constant 0 : i32
  %okA = comb.icmp eq %a, %eleven : i32
  %okB = comb.icmp eq %b, %zero : i32
  %okC = comb.icmp eq %c, %eleven : i32
  %okD = comb.icmp eq %d, %eleven : i32
  %ok = comb.and %okA, %okB, %okC, %okD : i1
  cf.cond_br %ok, ^pass, ^fail
^pass:
  %message = llvm.mlir.addressof @message : !llvm.ptr
  %printed = llvm.call @puts(%message) : (!llvm.ptr) -> i32
  return
^fail:
  llvm.call @abort() : () -> ()
  return
}
