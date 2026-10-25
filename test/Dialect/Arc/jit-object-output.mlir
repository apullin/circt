// RUN: arcilator %s --run --jit-entry=entry --jit-object-file=%t.o > %t.log 2>&1
// RUN: FileCheck %s --input-file=%t.log --check-prefix=CLEAN --allow-empty
// RUN: %PYTHON% -c "from pathlib import Path; assert Path(r'%t.o').stat().st_size > 0"

// Dumping before entry lookup tries to materialize a packed wrapper for the
// unused internal helper. Such a wrapper does not exist after optimization.
// CLEAN-NOT: Could not compile
llvm.func internal @unused_helper() {
  llvm.return
}
llvm.func @entry() {
  llvm.return
}
