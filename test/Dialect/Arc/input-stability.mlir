// RUN: circt-opt %s --arc-lower-state=infer-input-stability=true | FileCheck %s
// RUN: circt-opt %s --arc-lower-state | FileCheck %s --check-prefix=DEFAULT
// DEFAULT-NOT: arc.input_stable
// DEFAULT: arc.model
// DEFAULT-NOT: arc.input_stable

arc.define @identity(%d: i8) -> i8 {
  arc.output %d : i8
}
arc.define @bit(%d: i1) -> i1 {
  arc.output %d : i1
}
arc.define @nested(%d: i8) -> i8 {
  %v = arc.call @identity(%d) : (i8) -> i8
  %sum = comb.add %v, %d : i8
  arc.output %sum : i8
}
func.func private @foreign() -> i8
arc.define @unproved_arc(%clk: !seq.clock) -> !seq.clock {
  %v = seq.clock_inv %clk
  arc.output %v : !seq.clock
}

// Pure dataflow and pure nested calls are covered.
// CHECK-LABEL: arc.model @combinational
// CHECK-SAME: arc.input_stable
hw.module @combinational(in %a: i8, out y: i8) {
  %v = arc.call @nested(%a) : (i8) -> i8
  hw.output %v : i8
}

// Once the external clock is held fixed, no second clock event is possible.
// CHECK-LABEL: arc.model @clocked
// CHECK-SAME: arc.input_stable
hw.module @clocked(in %clk: !seq.clock, in %a: i8, out y: i8) {
  %v = arc.state @identity(%a) clock %clk latency 1 : (i8) -> i8
  hw.output %v : i8
}

// A state-derived clock needs a separate same-time propagation argument.
// CHECK-LABEL: arc.model @derived_clock
// CHECK-NOT: arc.input_stable
hw.module @derived_clock(in %clk: !seq.clock, in %a: i1, out y: i1) {
  %v = arc.state @bit(%a) clock %clk latency 1 : (i1) -> i1
  %derived = seq.to_clock %v
  %q = arc.state @bit(%a) clock %derived latency 1 : (i1) -> i1
  hw.output %q : i1
}

// Foreign calls are not assumed deterministic or effect-free.
// CHECK-LABEL: arc.model @foreign_call
// CHECK-NOT: arc.input_stable
hw.module @foreign_call(out y: i8) {
  %v = func.call @foreign() : () -> i8
  hw.output %v : i8
}

// Unsupported operations inside otherwise-pure arc definitions are also
// conservatively left unproved.
// CHECK-LABEL: arc.model @nested_unproved_call
// CHECK-NOT: arc.input_stable
hw.module @nested_unproved_call(in %clk: !seq.clock, out y: !seq.clock) {
  %v = arc.call @unproved_arc(%clk) : (!seq.clock) -> !seq.clock
  hw.output %v : !seq.clock
}

// Signal updates may intentionally settle over several evaluations.
// CHECK-LABEL: arc.model @mutable_signal
// CHECK-NOT: arc.input_stable
hw.module @mutable_signal(in %value: i8, in %enable: i1, out y: i8) {
  %zero = hw.constant 0 : i8
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %signal = llhd.sig %zero : <i8>
  %old = llhd.prb %signal : i8
  llhd.drv %signal, %value after %eps if %enable : i8
  hw.output %old : i8
}

// Aggregate input/state ownership is outside this scalar proof.
// CHECK-LABEL: arc.model @aggregate_port
// CHECK-NOT: arc.input_stable
hw.module @aggregate_port(in %a: !hw.array<2xi8>, out y: !hw.array<2xi8>) {
  hw.output %a : !hw.array<2xi8>
}

// Even an unused time read is deliberately not covered by the proof.
// CHECK-LABEL: arc.model @time_dependent
// CHECK-NOT: arc.input_stable
hw.module @time_dependent(in %a: i8, out y: i8) {
  %time = llhd.current_time
  hw.output %a : i8
}
