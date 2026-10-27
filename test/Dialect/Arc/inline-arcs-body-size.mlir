// RUN: circt-opt %s --arc-inline | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-opt %s --arc-inline="max-body-ops=2" | FileCheck %s --check-prefix=SMALL

// The limit counts non-constant operations in the body, excluding arc.define
// itself. Use each arc twice to avoid the independent single-use heuristic.
// DEFAULT-LABEL: hw.module @bodySize
// DEFAULT-NOT: arc.call @threeOps
// DEFAULT: arc.call @fourOps
// DEFAULT: arc.call @fourOps
// DEFAULT: hw.output
// DEFAULT-NOT: arc.define @threeOps
// DEFAULT: arc.define @fourOps
// SMALL-LABEL: hw.module @bodySize
// SMALL: arc.call @threeOps
// SMALL: arc.call @threeOps
// SMALL: arc.call @fourOps
// SMALL: arc.call @fourOps
// SMALL: hw.output
// SMALL: arc.define @threeOps
// SMALL: arc.define @fourOps
hw.module @bodySize(in %a: i4, in %b: i4, out x: i4, out y: i4,
                    out z: i4, out w: i4) {
  %0 = arc.call @threeOps(%a, %b) : (i4, i4) -> i4
  %1 = arc.call @threeOps(%a, %b) : (i4, i4) -> i4
  %2 = arc.call @fourOps(%a, %b) : (i4, i4) -> i4
  %3 = arc.call @fourOps(%a, %b) : (i4, i4) -> i4
  hw.output %0, %1, %2, %3 : i4, i4, i4, i4
}

arc.define @threeOps(%a: i4, %b: i4) -> i4 {
  %ones = hw.constant -1 : i4
  %0 = comb.xor %a, %ones : i4
  %1 = comb.xor %b, %ones : i4
  %2 = comb.or %0, %1 : i4
  arc.output %2 : i4
}

arc.define @fourOps(%a: i4, %b: i4) -> i4 {
  %ones = hw.constant -1 : i4
  %0 = comb.xor %a, %ones : i4
  %1 = comb.xor %b, %ones : i4
  %2 = comb.or %0, %1 : i4
  %3 = comb.and %2, %a : i4
  arc.output %3 : i4
}
