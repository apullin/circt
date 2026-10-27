// RUN: split-file %s %t
// RUN: circt-verilog --ir-moore --mlir-print-debuginfo %t/main.sv -I%t | FileCheck %s
// REQUIRES: slang

// CHECK: %aa = moore.variable : <i32> loc(#[[A:loc[0-9]+]])
// CHECK: %bb = moore.variable : <i32> loc(#[[B:loc[0-9]+]])
// CHECK: moore.blocking_assign %aa, {{.*}} loc(#[[CALL_A:loc[0-9]+]])
// CHECK: moore.blocking_assign %bb, {{.*}} loc(#[[CALL_B:loc[0-9]+]])
// CHECK: moore.blocking_assign %aa, {{.*}} loc(#[[RANGE:loc[0-9]+]])
// Equal offsets in separate source buffers must retain distinct locations.
// CHECK-DAG: #[[A]] = loc("a.sv":40:5)
// CHECK-DAG: #[[B]] = loc("b.sv":40:5)
// Macro expansions and multi-line ranges must retain their original mappings.
// CHECK-DAG: #[[CALL_A]] = loc("use_a.sv":200:0)
// CHECK-DAG: #[[CALL_B]] = loc("use_b.sv":300:0)
// CHECK-DAG: #[[RANGE]] = loc("range.sv":400:5 to 403:6)

//--- a.svh
`line 40 "a.sv" 0
int aa;

//--- b.svh
`line 40 "b.sv" 0
int bb;

//--- main.sv
`define ASSIGN(dst, rhs) dst = rhs;
module Locations;
  `include "a.svh"
  `include "b.svh"
  initial begin
`line 200 "use_a.sv" 0
    `ASSIGN(aa, 11)
`line 300 "use_b.sv" 0
    `ASSIGN(bb, 22)
`line 400 "range.sv" 0
    aa = (
      bb +
      1
    );
  end
endmodule
