// RUN: circt-opt %s --arc-infer-state-properties | FileCheck %s

arc.define @ArrayEnable(%enable: i1, %input: !hw.array<2xi8>, %old: !hw.array<2xi8>) -> !hw.array<2xi8> {
  %next = comb.mux %enable, %input, %old : !hw.array<2xi8>
  arc.output %next : !hw.array<2xi8>
}

// CHECK-LABEL: hw.module @AggregateEnable
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[ZERO:%.+]] = hw.constant 0 : i16
// CHECK: [[ARRAY:%.+]] = hw.bitcast [[ZERO]] : (i16) -> !hw.array<2xi8>
// CHECK: arc.state @ArrayEnable([[TRUE]], %input, [[ARRAY]]) clock %clock enable %enable latency 1
hw.module @AggregateEnable(in %clock: !seq.clock, in %enable: i1, in %input: !hw.array<2xi8>, out result: !hw.array<2xi8>) {
  %next = arc.state @ArrayEnable(%enable, %input, %next) clock %clock latency 1 : (i1, !hw.array<2xi8>, !hw.array<2xi8>) -> !hw.array<2xi8>
  hw.output %next : !hw.array<2xi8>
}
