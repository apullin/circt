// RUN: arcilator %s --no-runtime --no-generate-driver --infer-input-stability --state-file=%t.on.json -o %t.on.ll
// RUN: FileCheck %s --check-prefix=ON < %t.on.json
// RUN: arcilator %s --no-runtime --no-generate-driver --state-file=%t.off.json -o %t.off.ll
// RUN: FileCheck %s --check-prefix=OFF < %t.off.json
// RUN: diff %t.on.ll %t.off.ll
// RUN: not arcilator %s --infer-input-stability 2>&1 | FileCheck %s --check-prefix=RUNTIME
// ON: "name": "stable"
// ON: "canSkipUnchangedInputs": true
// OFF: "name": "stable"
// OFF-NOT: canSkipUnchangedInputs
// RUNTIME: --infer-input-stability requires --no-runtime

hw.module @stable(in %a: i8, in %b: i8, out y: i8) {
  %v = comb.add %a, %b : i8
  hw.output %v : i8
}
