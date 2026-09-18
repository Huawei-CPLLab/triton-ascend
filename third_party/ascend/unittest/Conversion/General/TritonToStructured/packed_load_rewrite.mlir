// These tests describe the packed-memory layouts used by qweights in the
// Ascend backend.  The intent is to keep the IR shape/offset logic visible to
// future readers: each function constructs a logical 2-D matrix, applies the
// storage formula for the compact layout, and then verifies the rewrite returns
// to the original logical tensor form.
// RUN: triton-opt '--triton-to-structured=enable-packed-load-rewrite=true' --split-input-file %s | FileCheck %s

module {
  tt.func public @w3_qh(%base: !tt.ptr<i8>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>
    %five = arith.constant 5 : i32
    %thirtyone = arith.constant 31 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %sixteen = arith.constant 16 : i32
    %five_b = tt.splat %five : i32 -> tensor<16x128xi32>
    %thirtyone_b = tt.splat %thirtyone : i32 -> tensor<16x128xi32>
    %four_b = tt.splat %four : i32 -> tensor<16x128xi32>
    %eight_b = tt.splat %eight : i32 -> tensor<16x128xi32>
    %sixteen_b = tt.splat %sixteen : i32 -> tensor<16x128xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x128xi32>
    %in_group = arith.andi %cb, %thirtyone_b : tensor<16x128xi32>
    %sub = arith.shrsi %in_group, %eight_b : tensor<16x128xi32>
    %row_term = arith.muli %rb, %sixteen_b : tensor<16x128xi32>
    %group_term = arith.muli %group, %four_b : tensor<16x128xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x128xi32>
    %offset = arith.addi %offset0, %sub : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr : tensor<16x128x!tt.ptr<i8>>
    tt.return
  }
}

// W3-QH: the compact buffer is smaller than the logical matrix, so the pass
// must materialize a range that matches the compact storage and then reshape it
// back into the logical matrix view.
// CHECK-LABEL: tt.func public @w3_qh
// CHECK: tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
// CHECK: tt.load
// CHECK: tt.reshape
// CHECK: tt.broadcast

// -----

module {
  tt.func public @w3_qh_k1024(%base: !tt.ptr<i8>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<1024xi32> -> tensor<1x1024xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x1024xi32>
    %cb = tt.broadcast %cs : tensor<1x1024xi32> -> tensor<16x1024xi32>
    %five = arith.constant 5 : i32
    %thirtyone = arith.constant 31 : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %onehundredtwentyeight = arith.constant 128 : i32
    %five_b = tt.splat %five : i32 -> tensor<16x1024xi32>
    %thirtyone_b = tt.splat %thirtyone : i32 -> tensor<16x1024xi32>
    %four_b = tt.splat %four : i32 -> tensor<16x1024xi32>
    %eight_b = tt.splat %eight : i32 -> tensor<16x1024xi32>
    %row_b = tt.splat %onehundredtwentyeight : i32 -> tensor<16x1024xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x1024xi32>
    %in_group = arith.andi %cb, %thirtyone_b : tensor<16x1024xi32>
    %sub = arith.shrsi %in_group, %eight_b : tensor<16x1024xi32>
    %row_term = arith.muli %rb, %row_b : tensor<16x1024xi32>
    %group_term = arith.muli %group, %four_b : tensor<16x1024xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x1024xi32>
    %offset = arith.addi %offset0, %sub : tensor<16x1024xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x1024x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x1024x!tt.ptr<i8>>, tensor<16x1024xi32>
    %v = tt.load %ptr : tensor<16x1024x!tt.ptr<i8>>
    tt.return
  }
}

// W3-QH with a larger width keeps the same rewrite pattern but changes the
// compact physical size.  The important part is that the pass still recognizes
// the same offset formula for the larger packed matrix.
// CHECK-LABEL: tt.func public @w3_qh_k1024
// CHECK: tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>

// -----

module {
  tt.func public @w4(%base: !tt.ptr<i8>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cs = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rb = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x128xi32>
    %cb = tt.broadcast %cs : tensor<1x128xi32> -> tensor<16x128xi32>
    %five = arith.constant 5 : i32
    %fifteen = arith.constant 15 : i32
    %sixteen = arith.constant 16 : i32
    %sixtyfour = arith.constant 64 : i32
    %five_b = tt.splat %five : i32 -> tensor<16x128xi32>
    %fifteen_b = tt.splat %fifteen : i32 -> tensor<16x128xi32>
    %sixteen_b = tt.splat %sixteen : i32 -> tensor<16x128xi32>
    %sixtyfour_b = tt.splat %sixtyfour : i32 -> tensor<16x128xi32>
    %group = arith.shrsi %cb, %five_b : tensor<16x128xi32>
    %in_group = arith.andi %cb, %fifteen_b : tensor<16x128xi32>
    %row_term = arith.muli %rb, %sixtyfour_b : tensor<16x128xi32>
    %group_term = arith.muli %group, %sixteen_b : tensor<16x128xi32>
    %offset0 = arith.addi %row_term, %group_term : tensor<16x128xi32>
    %offset = arith.addi %offset0, %in_group : tensor<16x128xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<16x128x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<16x128x!tt.ptr<i8>>, tensor<16x128xi32>
    %v = tt.load %ptr : tensor<16x128x!tt.ptr<i8>>
    tt.return
  }
}

// W4 packs the logical matrix by a different formula; this branch confirms the
// rewrite still recognizes the alternative mapping and emits the compact load
// required for the physical buffer.
// CHECK-LABEL: tt.func public @w4
// CHECK: tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
// CHECK: tt.load

// -----

module {
  tt.func public @irregular(%base: !tt.ptr<i8>) {
    %offset = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<32x!tt.ptr<i8>>
    %ptr = tt.addptr %ptrs, %offset : tensor<32x!tt.ptr<i8>>, tensor<32xi32>
    %v = tt.load %ptr : tensor<32x!tt.ptr<i8>>
    tt.return
  }
}

// The last case verifies that irregular loads are intentionally left alone.  If
// the access pattern does not match one of the recognized packed layouts, the
// pass must not rewrite it.
// CHECK-LABEL: tt.func public @irregular
// CHECK: tt.load
// CHECK-NOT: tt.make_range {end = 256 : i32, start = 0 : i32}
