// RUN: split-file %s %t
// RUN: circt-verilog --ir-hw %t/good.sv -o %t/default.mlir
// RUN: circt-verilog --ir-hw --verify-each=false %t/good.sv -o %t/final.mlir
// RUN: diff %t/default.mlir %t/final.mlir
// RUN: FileCheck %s --check-prefix=GOOD < %t/final.mlir
// RUN: circt-verilog --ir-hw --verify-each=false --emit-bytecode %t/good.sv -o %t/final.bc
// RUN: circt-opt %t/final.bc | FileCheck %s --check-prefix=GOOD
// RUN: not circt-verilog --ir-hw --verify-each=false %t/bad.sv 2>&1 | FileCheck %s --check-prefix=BAD
// REQUIRES: slang

// Checking only final IR produces the same valid circuit. Source diagnostics
// remain enabled, including when emitting bytecode for another compiler stage.
// GOOD: hw.module @Add
// GOOD: comb.add
// GOOD: hw.output
// BAD: error: use of undeclared identifier 'nonexistent'

//--- good.sv
module Add(input logic [7:0] a, b, output logic [7:0] y);
  assign y = a + b;
endmodule

//--- bad.sv
`default_nettype none
module Bad(output logic y);
  assign y = nonexistent;
endmodule
