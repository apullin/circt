// RUN: circt-opt %s --seq-infer-clock-ports | FileCheck %s
// RUN: circt-opt %s --seq-infer-clock-ports="promote-public-ports=false" | FileCheck %s --check-prefix=NOPUB
// RUN: circt-opt %s --seq-infer-clock-ports | circt-opt --seq-infer-clock-ports | FileCheck %s

// Mixed data widths must remain data while two clock ports and two forwarded
// clock results are retyped. The analysis bit vectors must outlive iteration;
// `.lookup(target).set_bits()` iterates a destroyed temporary in C++17.
// CHECK-LABEL: hw.module private @Leaf(in %clk0 : !seq.clock, in %data : i16, in %clk1 : !seq.clock, in %addr : i5
// CHECK-SAME: out c0 : !seq.clock, out d : i16, out c1 : !seq.clock, out a : i5
// NOPUB-LABEL: hw.module private @Leaf(in %clk0 : !seq.clock, in %data : i16, in %clk1 : !seq.clock, in %addr : i5
hw.module private @Leaf(in %clk0 : i1, in %data : i16, in %clk1 : i1, in %addr : i5,
                       out c0 : i1, out d : i16, out c1 : i1, out a : i5) {
  %c0 = seq.to_clock %clk0
  %c1 = seq.to_clock %clk1
  %d = seq.firreg %data clock %c0 : i16
  %a = seq.firreg %addr clock %c1 : i5
  hw.output %clk0, %d, %clk1, %a : i1, i16, i1, i5
}

// CHECK-LABEL: hw.module @Top(in %clk0 : i1, in %clk1 : i1, in %en : i1, in %data : i16, in %addr : i5
// CHECK-NOT: seq.to_clock %data
// CHECK-NOT: seq.to_clock %addr
// CHECK: [[GATE0:%.+]] = comb.xor %clk0, %en : i1
// CHECK: [[GATE1:%.+]] = comb.xor %clk1, %en : i1
// CHECK: [[C0:%.+]] = seq.to_clock [[GATE0]]
// CHECK: [[C1:%.+]] = seq.to_clock [[GATE1]]
// CHECK: hw.instance "leaf" @Leaf(clk0: [[C0]]: !seq.clock, data: %data: i16, clk1: [[C1]]: !seq.clock, addr: %addr: i5) -> (c0: !seq.clock, d: i16, c1: !seq.clock, a: i5)
// NOPUB-LABEL: hw.module @Top(in %clk0 : i1, in %clk1 : i1, in %en : i1, in %data : i16, in %addr : i5
// NOPUB: hw.instance "leaf" @Leaf(clk0: {{%.+}}: !seq.clock, data: %data: i16, clk1: {{%.+}}: !seq.clock, addr: %addr: i5) -> (c0: !seq.clock, d: i16, c1: !seq.clock, a: i5)
hw.module @Top(in %clk0 : i1, in %clk1 : i1, in %en : i1, in %data : i16, in %addr : i5,
               out q : i16, out a : i5) {
  %c0 = comb.xor %clk0, %en : i1
  %c1 = comb.xor %clk1, %en : i1
  %f0, %d, %f1, %a = hw.instance "leaf" @Leaf(clk0: %c0: i1, data: %data: i16, clk1: %c1: i1, addr: %addr: i5) -> (c0: i1, d: i16, c1: i1, a: i5)
  %clock0 = seq.to_clock %f0
  %clock1 = seq.to_clock %f1
  %q = seq.firreg %d clock %clock0 : i16
  %r = seq.firreg %a clock %clock1 : i5
  hw.output %q, %r : i16, i5
}
