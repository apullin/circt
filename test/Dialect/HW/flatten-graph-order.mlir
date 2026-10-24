// RUN: circt-opt %s --hw-flatten-modules | FileCheck %s

// Forward references must survive cloning, and unrelated parent operations
// must retain their positions between two uses of the same callee.
// CHECK-LABEL: hw.module @PreserveOrder
// CHECK-NOT: hw.instance
// CHECK: comb.xor
// CHECK-NEXT: hw.constant 3 : i8
// CHECK-NEXT: hw.constant 7 : i8
// CHECK-NEXT: comb.xor
// CHECK-NEXT: hw.constant 3 : i8
// CHECK-NEXT: comb.add
hw.module @PreserveOrder(in %x: i8, out a: i8, out b: i8) {
  %first = hw.instance "first" @ForwardReference(x: %x: i8) -> (y: i8)
  %unrelated = hw.constant 7 : i8
  %second = hw.instance "second" @ForwardReference(x: %x: i8) -> (y: i8)
  %sum = comb.add %first, %unrelated : i8
  hw.output %sum, %second : i8, i8
}
hw.module private @ForwardReference(in %x: i8, out y: i8) {
  %value = comb.xor %later, %x : i8
  %later = hw.constant 3 : i8
  hw.output %value : i8
}

// A callee containing only hw.output inserts no operations, both with a
// non-empty suffix and immediately before the parent terminator.
// CHECK-LABEL: hw.module @OutputOnlyAnchors
// CHECK-NOT: hw.instance
// CHECK: %[[C:.*]] = hw.constant 5 : i8
// CHECK-NEXT: hw.output %x, %[[C]] : i8, i8
hw.module @OutputOnlyAnchors(in %x: i8, out a: i8, out b: i8) {
  %first = hw.instance "first" @Identity(x: %x: i8) -> (y: i8)
  %unrelated = hw.constant 5 : i8
  %last = hw.instance "last" @Identity(x: %unrelated: i8) -> (y: i8)
  hw.output %first, %last : i8, i8
}
hw.module private @Identity(in %x: i8, out y: i8) {
  hw.output %x : i8
}
