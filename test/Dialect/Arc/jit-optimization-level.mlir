// RUN: arcilator %s --run --jit-entry=main --jit-opt-level=0
// RUN: arcilator %s --run --jit-entry=main --jit-opt-level=1
// RUN: arcilator %s --run --jit-entry=main --jit-opt-level=2
// RUN: arcilator %s --run --jit-entry=main --jit-opt-level=3
// RUN: not arcilator %s --run --jit-entry=main --jit-opt-level=4 2>&1 | FileCheck %s
// CHECK: --jit-opt-level must be between 0 and 3
llvm.func @main() {
  llvm.return
}
