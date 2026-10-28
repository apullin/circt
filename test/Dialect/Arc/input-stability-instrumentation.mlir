// RUN: circt-opt %s --arc-insert-runtime | FileCheck %s
// CHECK-NOT: arc.input_stable
// CHECK: arc.model @stable
// CHECK-NOT: arc.input_stable

// Instrumentation invalidates the property even if the underlying computation
// has no data-dependent state updates.
arc.model @stable io !hw.modty<input a: i8, output y: i8> storageBytes 2
    attributes {arc.input_stable} {
^bb0(%storage: !arc.storage):
  %a = arc.root_input "a", %storage {offset = 0 : i32} : (!arc.storage) -> !arc.state<i8>
  %y = arc.root_output "y", %storage {offset = 1 : i32} : (!arc.storage) -> !arc.state<i8>
  %v = arc.state_read %a : <i8>
  arc.state_write %y = %v : <i8>
}
