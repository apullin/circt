// RUN: circt-opt %s --canonicalize | circt-opt | FileCheck %s

arc.define @array_identity(%value: !hw.array<2xi8>) -> !hw.array<2xi8> {
  arc.output %value : !hw.array<2xi8>
}
arc.define @struct_identity(%value: !hw.struct<wide: i8, narrow: i3>) -> !hw.struct<wide: i8, narrow: i3> {
  arc.output %value : !hw.struct<wide: i8, narrow: i3>
}

// Aggregate initial zero values must remain valid aggregate attributes, rather
// than IntegerAttrs whose type is an array or struct. Require a parse roundtrip.
// CHECK-LABEL: @DisabledArray
// CHECK: hw.aggregate_constant [0 : i8, 0 : i8]
// CHECK-NOT: hw.param.value
// CHECK-NOT: arc.state
// CHECK: hw.output
hw.module @DisabledArray(in %clock: !seq.clock, in %input: !hw.array<2xi8>, out result: !hw.array<2xi8>) {
  %false = hw.constant false
  %state = arc.state @array_identity(%input) clock %clock enable %false latency 1 : (!hw.array<2xi8>) -> !hw.array<2xi8>
  hw.output %state : !hw.array<2xi8>
}

// CHECK-LABEL: @ResetStruct
// CHECK: hw.aggregate_constant [0 : i8, 0 : i3]
// CHECK-NOT: hw.param.value
// CHECK-NOT: arc.state
// CHECK: hw.output
hw.module @ResetStruct(in %clock: !seq.clock, in %input: !hw.struct<wide: i8, narrow: i3>, out result: !hw.struct<wide: i8, narrow: i3>) {
  %true = hw.constant true
  %state = arc.state @struct_identity(%input) clock %clock reset %true latency 1 : (!hw.struct<wide: i8, narrow: i3>) -> !hw.struct<wide: i8, narrow: i3>
  hw.output %state : !hw.struct<wide: i8, narrow: i3>
}

arc.define @mixed_identity(%array: !hw.array<2xi8>, %other: !hw.union<a: i8, b: i16>) -> (!hw.array<2xi8>, !hw.union<a: i8, b: i16>) {
  arc.output %array, %other : !hw.array<2xi8>, !hw.union<a: i8, b: i16>
}

// Leave all results intact when one type has no materializable zero attribute.
// CHECK-LABEL: @UnsupportedZero
// CHECK: arc.state @mixed_identity
// CHECK-NOT: hw.param.value
// CHECK: hw.output
hw.module @UnsupportedZero(in %clock: !seq.clock, in %array: !hw.array<2xi8>, in %other: !hw.union<a: i8, b: i16>, out array: !hw.array<2xi8>, out other: !hw.union<a: i8, b: i16>) {
  %false = hw.constant false
  %state:2 = arc.state @mixed_identity(%array, %other) clock %clock enable %false latency 1 : (!hw.array<2xi8>, !hw.union<a: i8, b: i16>) -> (!hw.array<2xi8>, !hw.union<a: i8, b: i16>)
  hw.output %state#0, %state#1 : !hw.array<2xi8>, !hw.union<a: i8, b: i16>
}
