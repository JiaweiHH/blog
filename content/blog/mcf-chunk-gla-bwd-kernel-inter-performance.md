+++
title = "【Q4-MCF】chunk_gla_bwd_kernel_inter 性能分析"
date = "2026-10-31T11:00:00+08:00"
slug = "mcf-chunk-gla-bwd-kernel-inter-performance"
description = "分析 chunk_gla_bwd_kernel_inter 在 MCF 开关下的 UB 地址复用、跨迭代阻塞依赖，以及 GSS 同步等待的优化思路。"
tags = ["编译器", "性能分析"]
draft = false
+++

在 mcf-on 版本中，性能劣化原因是 PlanMemory 的地址复用关系导致主 kernel 迭代间出现 ub 依赖，影响流水阻塞

## mcf-on 地址复用与阻塞依赖

```mlir
func.func @chunk_gla_bwd_kernel_inter_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg13: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg14: f32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<19xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c49664_i64 = arith.constant 49664 : i64
    %c33280_i64 = arith.constant 33280 : i64
    %c205568_i64 = arith.constant 205568 : i64
    %c221952_i64 = arith.constant 221952 : i64
    %c172800_i64 = arith.constant 172800 : i64
    %c189184_i64 = arith.constant 189184 : i64
    %c74496_i64 = arith.constant 74496 : i64
    %c90880_i64 = arith.constant 90880 : i64
    %c140032_i64 = arith.constant 140032 : i64
    %c156416_i64 = arith.constant 156416 : i64
    %c123648_i64 = arith.constant 123648 : i64
    %c131840_i64 = arith.constant 131840 : i64
    %c107264_i64 = arith.constant 107264 : i64
    %c115456_i64 = arith.constant 115456 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c33024_i64 = arith.constant 33024 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c3_i64 = arith.constant 3 : i64
    %c5_i64 = arith.constant 5 : i64
    %c64_i32 = arith.constant 64 : i32
    %c4_i64 = arith.constant 4 : i64
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c63_i32 = arith.constant 63 : i32
    %c28_i32 = arith.constant 28 : i32
    %c238336_i64 = arith.constant 238336 : i64
    %c238592_i64 = arith.constant 238592 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c41472_i64 = arith.constant 41472 : i64
    %c57856_i64 = arith.constant 57856 : i64
    %c66048_i64 = arith.constant 66048 : i64
    %c70272_i64 = arith.constant 70272 : i64
    %c238848_i64 = arith.constant 238848 : i64
    %c239104_i64 = arith.constant 239104 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca[%c0] : memref<1xi64>
    %0 = hivm.hir.pointer_cast(%c33280_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c49664_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c205568_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c221952_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c172800_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c189184_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c74496_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c90880_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c140032_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c156416_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c123648_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c131840_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c107264_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c115456_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c32768_i64) : memref<64xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c33024_i64) : memref<64xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c0_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c16384_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %alloca_1 = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca_1[%c0] : memref<1xi64>
    %18 = arith.muli %arg16, %arg17 : i32
    %19 = arith.muli %18, %arg18 : i32
    %20 = hivm.hir.get_block_idx -> i64
    %21 = arith.trunci %20 : i64 to i32
    %22 = hivm.hir.pointer_cast(%c238336_i64) : memref<64xf32, #hivm.address_space<ub>>
    call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_20(%22) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
    %23 = arith.extsi %arg15 : i32 to i64
    %24 = arith.addi %arg15, %c63_i32 : i32
    %25 = arith.divsi %24, %c64_i32 : i32
    %26 = arith.extsi %25 : i32 to i64
    %27 = arith.index_cast %arg15 : i32 to index
    %28 = hivm.hir.get_sub_block_idx -> i64
    %29 = arith.index_cast %28 : i64 to index
    %30 = arith.cmpi eq, %29, %c0 : index
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 7
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 7
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 9
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 9
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg19 = %21 to %19 step %c28_i32  : i32 {
      %31 = memref.load %alloca[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
      %32 = arith.remui %31, %c2_i64 : i64
      %33 = arith.index_cast %32 : i64 to index
      %34 = arith.cmpi eq, %32, %c1_i64 : i64
      %35 = arith.select %34, %1, %0 : memref<64x64xf32, #hivm.address_space<ub>>
      %36 = arith.select %34, %3, %2 : memref<64x64xf32, #hivm.address_space<ub>>
      %37 = arith.select %34, %5, %4 : memref<64x64xf32, #hivm.address_space<ub>>
      %38 = arith.select %34, %7, %6 : memref<64x64xf32, #hivm.address_space<ub>>
      %39 = arith.select %34, %9, %8 : memref<64x64xf32, #hivm.address_space<ub>>
      %40 = arith.select %34, %11, %10 : memref<64x64xbf16, #hivm.address_space<ub>>
      %41 = arith.select %34, %13, %12 : memref<64x64xbf16, #hivm.address_space<ub>>
      %42 = arith.select %34, %15, %14 : memref<64xf32, #hivm.address_space<ub>>
      %43 = arith.select %34, %17, %16 : memref<64x64xf32, #hivm.address_space<ub>>
      %44 = arith.index_cast %33 : index to i1
      %45 = arith.select %44, %c4_i64, %c5_i64 : i64
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %46 = arith.remsi %arg19, %arg16 : i32
      %47 = arith.divsi %arg19, %arg16 : i32
      %48 = arith.remsi %47, %arg17 : i32
      %49 = arith.divsi %arg19, %18 : i32
      %50 = arith.remsi %49, %arg18 : i32
      %51 = arith.extsi %48 : i32 to i64
      %52 = arith.extsi %50 : i32 to i64
      %53 = arith.divsi %52, %c4_i64 : i64
      %54 = arith.remsi %52, %c4_i64 : i64
      %55 = arith.muli %53, %26 : i64
      %56 = arith.addi %55, %51 : i64
      %57 = arith.muli %53, %23 : i64
      %58 = arith.muli %46, %c64_i32 : i32
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.muli %57, %c4_i64 : i64
      %61 = arith.addi %60, %54 : i64
      %62 = arith.muli %61, %c64_i64 : i64
      %63 = arith.index_cast %62 : i64 to index
      %64 = arith.muli %56, %c4_i64 : i64
      %65 = arith.addi %64, %54 : i64
      %66 = arith.muli %65, %c4096_i64 : i64
      %67 = arith.index_cast %66 : i64 to index
      %68 = arith.muli %51, %c64_i64 : i64
      %69 = arith.index_cast %68 : i64 to index
      %70 = affine.apply #map11()[%59, %63, %69]
      %71 = affine.apply #map4()[%69]
      %72 = arith.maxsi %69, %27 : index
      %73 = arith.minsi %71, %72 : index
      %74 = affine.apply #map5()[%73, %69]
      %75 = affine.apply #map4()[%59]
      %76 = arith.maxsi %59, %c64 : index
      %77 = arith.minsi %75, %76 : index
      %78 = affine.apply #map5()[%77, %59]
      %79 = arith.minsi %74, %c64 : index
      %80 = arith.maxsi %79, %c0 : index
      %81 = arith.minsi %78, %c64 : index
      %82 = arith.maxsi %81, %c0 : index
      %83 = arith.cmpi slt, %80, %c64 : index
      %84 = arith.cmpi slt, %82, %c64 : index
      %85 = arith.ori %83, %84 : i1
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg5 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%70], sizes: [%80, %82], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %43 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [0], sizes: [%80, %82], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %85 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_0(%43) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %45]
      %cast = memref.cast %reinterpret_cast : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_7 = memref.cast %reinterpret_cast_6 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast, %cast_7, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      %86 = arith.addi %68, %c64_i64 : i64
      %87 = arith.minsi %23, %86 : i64
      %88 = arith.subi %87, %c1_i64 : i64
      %89 = arith.muli %88, %c256_i64 : i64
      %90 = arith.addi %62, %89 : i64
      %91 = arith.index_cast %90 : i64 to index
      %92 = affine.apply #map12()[%91, %59]
      %93 = arith.cmpi slt, %78, %c64 : index
      %base_buffer_8, %offset_9, %sizes_10, %strides_11 = memref.extract_strided_metadata %arg5 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_12 = memref.reinterpret_cast %base_buffer_8 to offset: [%92], sizes: [%78], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %42 : memref<64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [0], sizes: [%78], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %93 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_1(%42) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      %cast_18 = memref.cast %reinterpret_cast_12 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_19 = memref.cast %reinterpret_cast_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_18, %cast_19, %c2_i32, %cst, %c0, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %94 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      %cast_20 = memref.cast %22 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %94 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_20, %cast_21) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      %95 = memref.load %alloca_1[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
      %96 = arith.remui %95, %c2_i64 : i64
      %97 = arith.index_cast %96 : i64 to index
      %98 = arith.index_cast %97 : index to i1
      %99 = arith.select %98, %c2_i64, %c3_i64 : i64
      %100 = affine.apply #map13()[%97]
      %101 = arith.index_cast %100 : index to i64
      %102 = affine.apply #map14()[%97]
      %103 = arith.index_cast %102 : index to i64
      %104 = arith.cmpi eq, %96, %c1_i64 : i64
      %105 = arith.select %104, %c57856_i64, %c49664_i64 : i64
      %106 = hivm.hir.pointer_cast(%105) : memref<32x64xf32, #hivm.address_space<ub>>
      %107 = arith.select %104, %c41472_i64, %c33280_i64 : i64
      %108 = hivm.hir.pointer_cast(%107) : memref<32x64xf32, #hivm.address_space<ub>>
      %109 = arith.select %104, %c20480_i64, %c4096_i64 : i64
      %110 = hivm.hir.pointer_cast(%109) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %111 = arith.select %104, %c16384_i64, %c0_i64 : i64
      %112 = hivm.hir.pointer_cast(%111) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %113 = affine.apply #map15()[%67, %59]
      %base_buffer_22, %offset_23, %sizes_24, %strides_25 = memref.extract_strided_metadata %arg6 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_26 = memref.reinterpret_cast %base_buffer_22 to offset: [%113], sizes: [32, %82], strides: [1, 64] : memref<f32, #hivm.address_space<gm>> to memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
      scf.if %84 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_2(%108) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %99]
      %cast_27 = memref.cast %reinterpret_cast_26 : memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_28 = memref.cast %108 : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_27, %cast_28, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      %base_buffer_29, %offset_30, %sizes_31, %strides_32 = memref.extract_strided_metadata %arg8 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_33 = memref.reinterpret_cast %base_buffer_29 to offset: [%113], sizes: [32, %82], strides: [1, 64] : memref<f32, #hivm.address_space<gm>> to memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
      scf.if %84 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_3(%106) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_34 = memref.cast %reinterpret_cast_33 : memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_35 = memref.cast %106 : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_34, %cast_35, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %114 = hivm.hir.pointer_cast(%c66048_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
      %base_buffer_36, %offset_37, %sizes_38:4, %strides_39:4 = memref.extract_strided_metadata %114 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_40 = memref.reinterpret_cast %base_buffer_36 to offset: [0], sizes: [4, 32, 16], strides: [528, 16, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
      %115 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      %116 = hivm.hir.pointer_cast(%c70272_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
      %base_buffer_41, %offset_42, %sizes_43:4, %strides_44:4 = memref.extract_strided_metadata %116 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_45 = memref.reinterpret_cast %base_buffer_41 to offset: [0], sizes: [4, 32, 16], strides: [528, 16, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_merged_vf_0(%108, %106, %reinterpret_cast_40, %94, %115, %106, %reinterpret_cast_45) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<32x64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %99]
      %base_buffer_46, %offset_47, %sizes_48:4, %strides_49:4 = memref.extract_strided_metadata %114 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %101
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %30 {
        %reinterpret_cast_167 = memref.reinterpret_cast %base_buffer_46 to offset: [0], sizes: [4, 512], strides: [528, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
        %base_buffer_168, %offset_169, %sizes_170:4, %strides_171:4 = memref.extract_strided_metadata %110 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
        %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_168 to offset: [0], sizes: [4, 512], strides: [512, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<4x512xbf16, #hivm.address_space<cbuf>>
        %cast_173 = memref.cast %reinterpret_cast_167 : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_174 = memref.cast %reinterpret_cast_172 : memref<4x512xbf16, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast_173, %cast_174) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      } {limit_sub_block_id0}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
      %base_buffer_50, %offset_51, %sizes_52:4, %strides_53:4 = memref.extract_strided_metadata %116 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %103
      scf.if %30 {
        %reinterpret_cast_167 = memref.reinterpret_cast %base_buffer_50 to offset: [0], sizes: [4, 512], strides: [528, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
        %base_buffer_168, %offset_169, %sizes_170:4, %strides_171:4 = memref.extract_strided_metadata %112 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
        %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_168 to offset: [0], sizes: [4, 512], strides: [512, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<4x512xbf16, #hivm.address_space<cbuf>>
        %cast_173 = memref.cast %reinterpret_cast_167 : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_174 = memref.cast %reinterpret_cast_172 : memref<4x512xbf16, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast_173, %cast_174) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      } {limit_sub_block_id0}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      %117 = arith.addi %95, %c1_i64 : i64
      memref.store %117, %alloca_1[%c0] : memref<1xi64>
      %118 = arith.remui %117, %c2_i64 : i64
      %119 = arith.index_cast %118 : i64 to index
      %120 = arith.index_cast %119 : index to i1
      %121 = arith.select %120, %c2_i64, %c3_i64 : i64
      %122 = affine.apply #map13()[%119]
      %123 = arith.index_cast %122 : index to i64
      %124 = affine.apply #map14()[%119]
      %125 = arith.index_cast %124 : index to i64
      %126 = arith.cmpi eq, %118, %c1_i64 : i64
      %127 = arith.select %126, %c57856_i64, %c49664_i64 : i64
      %128 = hivm.hir.pointer_cast(%127) : memref<32x64xf32, #hivm.address_space<ub>>
      %129 = arith.select %126, %c41472_i64, %c33280_i64 : i64
      %130 = hivm.hir.pointer_cast(%129) : memref<32x64xf32, #hivm.address_space<ub>>
      %131 = arith.select %126, %c20480_i64, %c4096_i64 : i64
      %132 = hivm.hir.pointer_cast(%131) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %133 = arith.select %126, %c16384_i64, %c0_i64 : i64
      %134 = hivm.hir.pointer_cast(%133) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %135 = affine.apply #map16()[%67, %59]
      %base_buffer_54, %offset_55, %sizes_56, %strides_57 = memref.extract_strided_metadata %arg6 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_58 = memref.reinterpret_cast %base_buffer_54 to offset: [%135], sizes: [32, %82], strides: [1, 64] : memref<f32, #hivm.address_space<gm>> to memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
      scf.if %84 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_2(%130) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %121]
      %cast_59 = memref.cast %reinterpret_cast_58 : memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_60 = memref.cast %130 : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_59, %cast_60, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      %base_buffer_61, %offset_62, %sizes_63, %strides_64 = memref.extract_strided_metadata %arg8 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_65 = memref.reinterpret_cast %base_buffer_61 to offset: [%135], sizes: [32, %82], strides: [1, 64] : memref<f32, #hivm.address_space<gm>> to memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
      scf.if %84 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_3(%128) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_66 = memref.cast %reinterpret_cast_65 : memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_67 = memref.cast %128 : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_66, %cast_67, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %136 = hivm.hir.pointer_cast(%c66048_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
      %base_buffer_68, %offset_69, %sizes_70:4, %strides_71:4 = memref.extract_strided_metadata %136 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_72 = memref.reinterpret_cast %base_buffer_68 to offset: [0], sizes: [4, 32, 16], strides: [528, 16, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
      %137 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      %138 = hivm.hir.pointer_cast(%c70272_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
      %base_buffer_73, %offset_74, %sizes_75:4, %strides_76:4 = memref.extract_strided_metadata %138 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_77 = memref.reinterpret_cast %base_buffer_73 to offset: [0], sizes: [4, 32, 16], strides: [528, 16, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_merged_vf_0(%130, %128, %reinterpret_cast_72, %115, %137, %128, %reinterpret_cast_77) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<32x64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %121]
      %base_buffer_78, %offset_79, %sizes_80:4, %strides_81:4 = memref.extract_strided_metadata %136 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %123
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %30 {
        %reinterpret_cast_167 = memref.reinterpret_cast %base_buffer_78 to offset: [0], sizes: [4, 512], strides: [528, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
        %base_buffer_168, %offset_169, %sizes_170:4, %strides_171:4 = memref.extract_strided_metadata %132 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
        %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_168 to offset: [0], sizes: [4, 512], strides: [512, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<4x512xbf16, #hivm.address_space<cbuf>>
        %cast_173 = memref.cast %reinterpret_cast_167 : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_174 = memref.cast %reinterpret_cast_172 : memref<4x512xbf16, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast_173, %cast_174) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      } {limit_sub_block_id0}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
      %base_buffer_82, %offset_83, %sizes_84:4, %strides_85:4 = memref.extract_strided_metadata %138 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %125
      scf.if %30 {
        %reinterpret_cast_167 = memref.reinterpret_cast %base_buffer_82 to offset: [0], sizes: [4, 512], strides: [528, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
        %base_buffer_168, %offset_169, %sizes_170:4, %strides_171:4 = memref.extract_strided_metadata %134 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
        %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_168 to offset: [0], sizes: [4, 512], strides: [512, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<4x512xbf16, #hivm.address_space<cbuf>>
        %cast_173 = memref.cast %reinterpret_cast_167 : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_174 = memref.cast %reinterpret_cast_172 : memref<4x512xbf16, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast_173, %cast_174) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      } {limit_sub_block_id0}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      %139 = arith.addi %95, %c2_i64 : i64
      memref.store %139, %alloca_1[%c0] : memref<1xi64>
      %140 = arith.addi %95, %c3_i64 : i64
      memref.store %140, %alloca_1[%c0] : memref<1xi64>
      %141 = arith.addi %95, %c4_i64 : i64
      memref.store %141, %alloca_1[%c0] : memref<1xi64>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      %142 = hivm.hir.pointer_cast(%c238848_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_10(%137, %42, %142) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_11(%38, %arg14, %38) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, f32, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_12(%38, %43, %42, %38, %43) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_13(%39, %43, %43) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %base_buffer_86, %offset_87, %sizes_88, %strides_89 = memref.extract_strided_metadata %arg2 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_90 = memref.reinterpret_cast %base_buffer_86 to offset: [%70], sizes: [%80, %82], strides: [256, 1] : memref<bf16, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_91, %offset_92, %sizes_93:2, %strides_94:2 = memref.extract_strided_metadata %41 : memref<64x64xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_95 = memref.reinterpret_cast %base_buffer_91 to offset: [0], sizes: [%80, %82], strides: [64, 1] : memref<bf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %85 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_6(%41) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      %cast_96 = memref.cast %reinterpret_cast_90 : memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_97 = memref.cast %reinterpret_cast_95 : memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_bfloat16_t(%cast_96, %cast_97, %c2_i32, %cst_0, %c0, %c0_i32) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
      %base_buffer_98, %offset_99, %sizes_100, %strides_101 = memref.extract_strided_metadata %arg3 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_102 = memref.reinterpret_cast %base_buffer_98 to offset: [%70], sizes: [%80, %82], strides: [256, 1] : memref<bf16, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_103, %offset_104, %sizes_105:2, %strides_106:2 = memref.extract_strided_metadata %40 : memref<64x64xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_107 = memref.reinterpret_cast %base_buffer_103 to offset: [0], sizes: [%80, %82], strides: [64, 1] : memref<bf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %85 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_7(%40) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      %cast_108 = memref.cast %reinterpret_cast_102 : memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_109 = memref.cast %reinterpret_cast_107 : memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_bfloat16_t(%cast_108, %cast_109, %c2_i32, %cst_0, %c0, %c0_i32) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %143 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_14(%40, %143) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      %144 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_15(%43, %143, %144) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      %base_buffer_110, %offset_111, %sizes_112, %strides_113 = memref.extract_strided_metadata %arg9 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_114 = memref.reinterpret_cast %base_buffer_110 to offset: [%70], sizes: [%80, %82], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_115, %offset_116, %sizes_117:2, %strides_118:2 = memref.extract_strided_metadata %39 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_119 = memref.reinterpret_cast %base_buffer_115 to offset: [0], sizes: [%80, %82], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %85 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_8(%39) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_120 = memref.cast %reinterpret_cast_114 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_121 = memref.cast %reinterpret_cast_119 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_120, %cast_121, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 9
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_16(%38, %39, %38) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
      %base_buffer_122, %offset_123, %sizes_124, %strides_125 = memref.extract_strided_metadata %arg10 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_126 = memref.reinterpret_cast %base_buffer_122 to offset: [%70], sizes: [%80, %82], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_127, %offset_128, %sizes_129:2, %strides_130:2 = memref.extract_strided_metadata %37 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_131 = memref.reinterpret_cast %base_buffer_127 to offset: [0], sizes: [%80, %82], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %85 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_9(%37) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_132 = memref.cast %reinterpret_cast_126 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_133 = memref.cast %reinterpret_cast_131 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_132, %cast_133, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_17(%41, %38, %143, %43, %37, %37, %36) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %45]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
      %cast_134 = memref.cast %37 : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_135 = memref.cast %35 : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_136 = memref.cast %35 : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @cumsum_2d_float_dim0_comp(%cast_134, %cast_135, %cast_136, %false) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i1) -> ()
      %145 = hivm.hir.pointer_cast(%c239104_i64) : memref<64xf32, #hivm.address_space<ub>>
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_18(%37, %145) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_19(%37, %35, %145, %142, %144, %35) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %base_buffer_137, %offset_138, %sizes_139:2, %strides_140:2 = memref.extract_strided_metadata %38 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_141 = memref.reinterpret_cast %base_buffer_137 to offset: [0], sizes: [%80, %82], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %base_buffer_142, %offset_143, %sizes_144, %strides_145 = memref.extract_strided_metadata %arg11 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_146 = memref.reinterpret_cast %base_buffer_142 to offset: [%70], sizes: [%80, %82], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
      scf.if %30 {
        %cast_167 = memref.cast %reinterpret_cast_141 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_168 = memref.cast %reinterpret_cast_146 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_167, %cast_168, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {limit_sub_block_id0}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 7
      %base_buffer_147, %offset_148, %sizes_149:2, %strides_150:2 = memref.extract_strided_metadata %36 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_151 = memref.reinterpret_cast %base_buffer_147 to offset: [0], sizes: [%80, %82], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %base_buffer_152, %offset_153, %sizes_154, %strides_155 = memref.extract_strided_metadata %arg12 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_156 = memref.reinterpret_cast %base_buffer_152 to offset: [%70], sizes: [%80, %82], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      scf.if %30 {
        %cast_167 = memref.cast %reinterpret_cast_151 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_168 = memref.cast %reinterpret_cast_156 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_167, %cast_168, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {limit_sub_block_id0}
      %base_buffer_157, %offset_158, %sizes_159:2, %strides_160:2 = memref.extract_strided_metadata %35 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_161 = memref.reinterpret_cast %base_buffer_157 to offset: [0], sizes: [%80, %82], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %base_buffer_162, %offset_163, %sizes_164, %strides_165 = memref.extract_strided_metadata %arg13 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_166 = memref.reinterpret_cast %base_buffer_162 to offset: [%70], sizes: [%80, %82], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %30 {
        %cast_167 = memref.cast %reinterpret_cast_161 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_168 = memref.cast %reinterpret_cast_166 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_167, %cast_168, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {limit_sub_block_id0}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_ctrl true at ctrl[60]
      %146 = arith.addi %31, %c1_i64 : i64
      memref.store %146, %alloca[%c0] : memref<1xi64>
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 5
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 6
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
```

其中 `%35` 存在和 `%108, %130, %106, %128` 的地址复用，它们的地址选择如下：

```mlir
%0 = hivm.hir.pointer_cast(%c33280_i64) : memref<64x64xf32, #hivm.address_space<ub>>
%1 = hivm.hir.pointer_cast(%c49664_i64) : memref<64x64xf32, #hivm.address_space<ub>>
%35 = arith.select %34, %1, %0 : memref<64x64xf32, #hivm.address_space<ub>>

%107 = arith.select %104, %c41472_i64, %c33280_i64 : i64
%108 = hivm.hir.pointer_cast(%107) : memref<32x64xf32, #hivm.address_space<ub>>

%129 = arith.select %126, %c41472_i64, %c33280_i64 : i64
%130 = hivm.hir.pointer_cast(%129) : memref<32x64xf32, #hivm.address_space<ub>>

%105 = arith.select %104, %c57856_i64, %c49664_i64 : i64
%106 = hivm.hir.pointer_cast(%105) : memref<32x64xf32, #hivm.address_space<ub>>

%127 = arith.select %126, %c57856_i64, %c49664_i64 : i64
%128 = hivm.hir.pointer_cast(%127) : memref<32x64xf32, #hivm.address_space<ub>>
```

因此：

- %35 选择 %0 时，覆盖 %108 和 %130 的空间。
- %35 选择 %1 时，覆盖 %106 和 %128 的空间。

具体的阻塞依赖链为：

1. 迭代 `i` 的 `vf_19` store `%35` ，随后 `%35` 在 loop 末尾通过 `store_ubuf_to_gm_2d_float` 写入到 gm
2. 迭代 `i+1` 在 load `%108, %106, %130, %128` 的时候需要等待迭代 `i` 的 MTE3 结束，因此需要插入 `hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]`
3. `%108, %106, %130, %128` 加载完成之后 `outlined_merged_vf_0` 才能消费这些数据，推迟 L1 和 CUBE 的计算

```text
第 i 轮：%37 → cumsum → %35 → VF19 → %35 → dg 写回
                                          ↓
第 i+1 轮：加载 %108/%106/%130/%128 → 预处理 → 后续 CUBE
```

```text
UB 地址          33280          41472          49664          57856          66048
                  |--------------|--------------|--------------|--------------|
H/DH 阶段        |  H：%108      |  H：%130      |  DH：%106     |  DH：%128     |
                  |-----------------------------|-----------------------------|
cumsum 输出 /    |       %0：16 KiB             |       %1：16 KiB             |
最终 DG 阶段     |       [33280, 49664)          |       [49664, 66048)          |
                  \_____________________________ _____________________________/
                                                V
                                  %35 按外层迭代奇偶选择其中一个槽
```

## mcf-off 对比

```mlir
func.func @chunk_gla_bwd_kernel_inter_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg13: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg14: f32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<19xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c156416_i64 = arith.constant 156416 : i64
    %c140032_i64 = arith.constant 140032 : i64
    %c33280_i64 = arith.constant 33280 : i64
    %c49664_i64 = arith.constant 49664 : i64
    %c205568_i64 = arith.constant 205568 : i64
    %c221952_i64 = arith.constant 221952 : i64
    %c74496_i64 = arith.constant 74496 : i64
    %c90880_i64 = arith.constant 90880 : i64
    %c172800_i64 = arith.constant 172800 : i64
    %c189184_i64 = arith.constant 189184 : i64
    %c123648_i64 = arith.constant 123648 : i64
    %c131840_i64 = arith.constant 131840 : i64
    %c107264_i64 = arith.constant 107264 : i64
    %c115456_i64 = arith.constant 115456 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c33024_i64 = arith.constant 33024 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c3_i64 = arith.constant 3 : i64
    %c5_i64 = arith.constant 5 : i64
    %c64_i32 = arith.constant 64 : i32
    %c4_i64 = arith.constant 4 : i64
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c63_i32 = arith.constant 63 : i32
    %c28_i32 = arith.constant 28 : i32
    %c238336_i64 = arith.constant 238336 : i64
    %c238592_i64 = arith.constant 238592 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c41472_i64 = arith.constant 41472 : i64
    %c57856_i64 = arith.constant 57856 : i64
    %c66048_i64 = arith.constant 66048 : i64
    %c70272_i64 = arith.constant 70272 : i64
    %c238848_i64 = arith.constant 238848 : i64
    %c239104_i64 = arith.constant 239104 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca[%c0] : memref<1xi64>
    %0 = hivm.hir.pointer_cast(%c140032_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c156416_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c33280_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c49664_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c205568_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c221952_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c74496_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c90880_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c172800_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c189184_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c123648_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c131840_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c107264_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c115456_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c32768_i64) : memref<64xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c33024_i64) : memref<64xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c0_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c16384_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %alloca_1 = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca_1[%c0] : memref<1xi64>
    %18 = arith.muli %arg16, %arg17 : i32
    %19 = arith.muli %18, %arg18 : i32
    %20 = hivm.hir.get_block_idx -> i64
    %21 = arith.trunci %20 : i64 to i32
    %22 = hivm.hir.pointer_cast(%c238336_i64) : memref<64xf32, #hivm.address_space<ub>>
    call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_21(%22) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
    %23 = arith.extsi %arg15 : i32 to i64
    %24 = arith.addi %arg15, %c63_i32 : i32
    %25 = arith.divsi %24, %c64_i32 : i32
    %26 = arith.extsi %25 : i32 to i64
    %27 = arith.index_cast %arg15 : i32 to index
    %28 = hivm.hir.get_sub_block_idx -> i64
    %29 = arith.index_cast %28 : i64 to index
    %30 = arith.cmpi eq, %29, %c0 : index
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 6
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 6
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 8
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 8
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg19 = %21 to %19 step %c28_i32  : i32 {
      %31 = memref.load %alloca[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
      %32 = arith.remui %31, %c2_i64 : i64
      %33 = arith.index_cast %32 : i64 to index
      %34 = arith.cmpi eq, %32, %c1_i64 : i64
      %35 = arith.select %34, %1, %0 : memref<64x64xf32, #hivm.address_space<ub>>
      %36 = arith.select %34, %3, %2 : memref<64x64xf32, #hivm.address_space<ub>>
      %37 = arith.select %34, %5, %4 : memref<64x64xf32, #hivm.address_space<ub>>
      %38 = arith.select %34, %7, %6 : memref<64x64xf32, #hivm.address_space<ub>>
      %39 = arith.select %34, %9, %8 : memref<64x64xf32, #hivm.address_space<ub>>
      %40 = arith.select %34, %11, %10 : memref<64x64xbf16, #hivm.address_space<ub>>
      %41 = arith.select %34, %13, %12 : memref<64x64xbf16, #hivm.address_space<ub>>
      %42 = arith.select %34, %15, %14 : memref<64xf32, #hivm.address_space<ub>>
      %43 = arith.select %34, %17, %16 : memref<64x64xf32, #hivm.address_space<ub>>
      %44 = arith.index_cast %33 : index to i1
      %45 = arith.select %44, %c4_i64, %c5_i64 : i64
      %46 = arith.select %44, %c2_i64, %c3_i64 : i64
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %47 = arith.remsi %arg19, %arg16 : i32
      %48 = arith.divsi %arg19, %arg16 : i32
      %49 = arith.remsi %48, %arg17 : i32
      %50 = arith.divsi %arg19, %18 : i32
      %51 = arith.remsi %50, %arg18 : i32
      %52 = arith.extsi %49 : i32 to i64
      %53 = arith.extsi %51 : i32 to i64
      %54 = arith.divsi %53, %c4_i64 : i64
      %55 = arith.remsi %53, %c4_i64 : i64
      %56 = arith.muli %54, %26 : i64
      %57 = arith.addi %56, %52 : i64
      %58 = arith.muli %54, %23 : i64
      %59 = arith.muli %47, %c64_i32 : i32
      %60 = arith.index_cast %59 : i32 to index
      %61 = arith.muli %58, %c4_i64 : i64
      %62 = arith.addi %61, %55 : i64
      %63 = arith.muli %62, %c64_i64 : i64
      %64 = arith.index_cast %63 : i64 to index
      %65 = arith.muli %57, %c4_i64 : i64
      %66 = arith.addi %65, %55 : i64
      %67 = arith.muli %66, %c4096_i64 : i64
      %68 = arith.index_cast %67 : i64 to index
      %69 = arith.muli %52, %c64_i64 : i64
      %70 = arith.index_cast %69 : i64 to index
      %71 = affine.apply #map11()[%60, %64, %70]
      %72 = affine.apply #map4()[%70]
      %73 = arith.maxsi %70, %27 : index
      %74 = arith.minsi %72, %73 : index
      %75 = affine.apply #map5()[%74, %70]
      %76 = affine.apply #map4()[%60]
      %77 = arith.maxsi %60, %c64 : index
      %78 = arith.minsi %76, %77 : index
      %79 = affine.apply #map5()[%78, %60]
      %80 = arith.minsi %75, %c64 : index
      %81 = arith.maxsi %80, %c0 : index
      %82 = arith.minsi %79, %c64 : index
      %83 = arith.maxsi %82, %c0 : index
      %84 = arith.cmpi slt, %81, %c64 : index
      %85 = arith.cmpi slt, %83, %c64 : index
      %86 = arith.ori %84, %85 : i1
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg5 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%71], sizes: [%81, %83], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %43 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [0], sizes: [%81, %83], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %86 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_0(%43) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %45]
      %cast = memref.cast %reinterpret_cast : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_7 = memref.cast %reinterpret_cast_6 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast, %cast_7, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      %87 = arith.addi %69, %c64_i64 : i64
      %88 = arith.minsi %23, %87 : i64
      %89 = arith.subi %88, %c1_i64 : i64
      %90 = arith.muli %89, %c256_i64 : i64
      %91 = arith.addi %63, %90 : i64
      %92 = arith.index_cast %91 : i64 to index
      %93 = affine.apply #map12()[%92, %60]
      %94 = arith.cmpi slt, %79, %c64 : index
      %base_buffer_8, %offset_9, %sizes_10, %strides_11 = memref.extract_strided_metadata %arg5 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_12 = memref.reinterpret_cast %base_buffer_8 to offset: [%93], sizes: [%79], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %42 : memref<64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [0], sizes: [%79], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %94 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_1(%42) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      %cast_18 = memref.cast %reinterpret_cast_12 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_19 = memref.cast %reinterpret_cast_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_18, %cast_19, %c2_i32, %cst, %c0, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %95 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      %cast_20 = memref.cast %22 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %95 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_20, %cast_21) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %96 = memref.load %alloca_1[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
      %97 = arith.remui %96, %c2_i64 : i64
      %98 = arith.index_cast %97 : i64 to index
      %99 = arith.index_cast %98 : index to i1
      %100 = arith.select %99, %c2_i64, %c3_i64 : i64
      %101 = affine.apply #map13()[%98]
      %102 = arith.index_cast %101 : index to i64
      %103 = affine.apply #map14()[%98]
      %104 = arith.index_cast %103 : index to i64
      %105 = arith.cmpi eq, %97, %c1_i64 : i64
      %106 = arith.select %105, %c57856_i64, %c49664_i64 : i64
      %107 = hivm.hir.pointer_cast(%106) : memref<32x64xf32, #hivm.address_space<ub>>
      %108 = arith.select %105, %c41472_i64, %c33280_i64 : i64
      %109 = hivm.hir.pointer_cast(%108) : memref<32x64xf32, #hivm.address_space<ub>>
      %110 = arith.select %105, %c20480_i64, %c4096_i64 : i64
      %111 = hivm.hir.pointer_cast(%110) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %112 = arith.select %105, %c16384_i64, %c0_i64 : i64
      %113 = hivm.hir.pointer_cast(%112) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %114 = affine.apply #map15()[%68, %60]
      %base_buffer_22, %offset_23, %sizes_24, %strides_25 = memref.extract_strided_metadata %arg6 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_26 = memref.reinterpret_cast %base_buffer_22 to offset: [%114], sizes: [32, %83], strides: [1, 64] : memref<f32, #hivm.address_space<gm>> to memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
      scf.if %85 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_2(%109) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %100]
      %cast_27 = memref.cast %reinterpret_cast_26 : memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_28 = memref.cast %109 : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_27, %cast_28, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      %base_buffer_29, %offset_30, %sizes_31, %strides_32 = memref.extract_strided_metadata %arg8 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_33 = memref.reinterpret_cast %base_buffer_29 to offset: [%114], sizes: [32, %83], strides: [1, 64] : memref<f32, #hivm.address_space<gm>> to memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
      scf.if %85 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_3(%107) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_34 = memref.cast %reinterpret_cast_33 : memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_35 = memref.cast %107 : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_34, %cast_35, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %115 = hivm.hir.pointer_cast(%c66048_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
      %base_buffer_36, %offset_37, %sizes_38:4, %strides_39:4 = memref.extract_strided_metadata %115 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_40 = memref.reinterpret_cast %base_buffer_36 to offset: [0], sizes: [4, 32, 16], strides: [528, 16, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
      %116 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      %117 = hivm.hir.pointer_cast(%c70272_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
      %base_buffer_41, %offset_42, %sizes_43:4, %strides_44:4 = memref.extract_strided_metadata %117 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_45 = memref.reinterpret_cast %base_buffer_41 to offset: [0], sizes: [4, 32, 16], strides: [528, 16, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_merged_vf_0(%109, %107, %reinterpret_cast_40, %95, %116, %107, %reinterpret_cast_45) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<32x64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %100]
      %base_buffer_46, %offset_47, %sizes_48:4, %strides_49:4 = memref.extract_strided_metadata %115 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %102
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %30 {
        %reinterpret_cast_167 = memref.reinterpret_cast %base_buffer_46 to offset: [0], sizes: [4, 512], strides: [528, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
        %base_buffer_168, %offset_169, %sizes_170:4, %strides_171:4 = memref.extract_strided_metadata %111 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
        %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_168 to offset: [0], sizes: [4, 512], strides: [512, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<4x512xbf16, #hivm.address_space<cbuf>>
        %cast_173 = memref.cast %reinterpret_cast_167 : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_174 = memref.cast %reinterpret_cast_172 : memref<4x512xbf16, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast_173, %cast_174) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      } {limit_sub_block_id0}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
      %base_buffer_50, %offset_51, %sizes_52:4, %strides_53:4 = memref.extract_strided_metadata %117 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %104
      scf.if %30 {
        %reinterpret_cast_167 = memref.reinterpret_cast %base_buffer_50 to offset: [0], sizes: [4, 512], strides: [528, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
        %base_buffer_168, %offset_169, %sizes_170:4, %strides_171:4 = memref.extract_strided_metadata %113 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
        %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_168 to offset: [0], sizes: [4, 512], strides: [512, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<4x512xbf16, #hivm.address_space<cbuf>>
        %cast_173 = memref.cast %reinterpret_cast_167 : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_174 = memref.cast %reinterpret_cast_172 : memref<4x512xbf16, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast_173, %cast_174) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      } {limit_sub_block_id0}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      %118 = arith.addi %96, %c1_i64 : i64
      memref.store %118, %alloca_1[%c0] : memref<1xi64>
      %119 = arith.remui %118, %c2_i64 : i64
      %120 = arith.index_cast %119 : i64 to index
      %121 = arith.index_cast %120 : index to i1
      %122 = arith.select %121, %c2_i64, %c3_i64 : i64
      %123 = affine.apply #map13()[%120]
      %124 = arith.index_cast %123 : index to i64
      %125 = affine.apply #map14()[%120]
      %126 = arith.index_cast %125 : index to i64
      %127 = arith.cmpi eq, %119, %c1_i64 : i64
      %128 = arith.select %127, %c57856_i64, %c49664_i64 : i64
      %129 = hivm.hir.pointer_cast(%128) : memref<32x64xf32, #hivm.address_space<ub>>
      %130 = arith.select %127, %c41472_i64, %c33280_i64 : i64
      %131 = hivm.hir.pointer_cast(%130) : memref<32x64xf32, #hivm.address_space<ub>>
      %132 = arith.select %127, %c20480_i64, %c4096_i64 : i64
      %133 = hivm.hir.pointer_cast(%132) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %134 = arith.select %127, %c16384_i64, %c0_i64 : i64
      %135 = hivm.hir.pointer_cast(%134) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
      %136 = affine.apply #map16()[%68, %60]
      %base_buffer_54, %offset_55, %sizes_56, %strides_57 = memref.extract_strided_metadata %arg6 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_58 = memref.reinterpret_cast %base_buffer_54 to offset: [%136], sizes: [32, %83], strides: [1, 64] : memref<f32, #hivm.address_space<gm>> to memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
      scf.if %85 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_2(%131) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %122]
      %cast_59 = memref.cast %reinterpret_cast_58 : memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_60 = memref.cast %131 : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_59, %cast_60, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      %base_buffer_61, %offset_62, %sizes_63, %strides_64 = memref.extract_strided_metadata %arg8 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_65 = memref.reinterpret_cast %base_buffer_61 to offset: [%136], sizes: [32, %83], strides: [1, 64] : memref<f32, #hivm.address_space<gm>> to memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
      scf.if %85 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_3(%129) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_66 = memref.cast %reinterpret_cast_65 : memref<32x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_67 = memref.cast %129 : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_66, %cast_67, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %137 = hivm.hir.pointer_cast(%c66048_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
      %base_buffer_68, %offset_69, %sizes_70:4, %strides_71:4 = memref.extract_strided_metadata %137 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_72 = memref.reinterpret_cast %base_buffer_68 to offset: [0], sizes: [4, 32, 16], strides: [528, 16, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
      %138 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      %139 = hivm.hir.pointer_cast(%c70272_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
      %base_buffer_73, %offset_74, %sizes_75:4, %strides_76:4 = memref.extract_strided_metadata %139 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_77 = memref.reinterpret_cast %base_buffer_73 to offset: [0], sizes: [4, 32, 16], strides: [528, 16, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_merged_vf_0(%131, %129, %reinterpret_cast_72, %116, %138, %129, %reinterpret_cast_77) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<32x64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %122]
      %base_buffer_78, %offset_79, %sizes_80:4, %strides_81:4 = memref.extract_strided_metadata %137 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %124
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %30 {
        %reinterpret_cast_167 = memref.reinterpret_cast %base_buffer_78 to offset: [0], sizes: [4, 512], strides: [528, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
        %base_buffer_168, %offset_169, %sizes_170:4, %strides_171:4 = memref.extract_strided_metadata %133 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
        %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_168 to offset: [0], sizes: [4, 512], strides: [512, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<4x512xbf16, #hivm.address_space<cbuf>>
        %cast_173 = memref.cast %reinterpret_cast_167 : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_174 = memref.cast %reinterpret_cast_172 : memref<4x512xbf16, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast_173, %cast_174) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      } {limit_sub_block_id0}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
      %base_buffer_82, %offset_83, %sizes_84:4, %strides_85:4 = memref.extract_strided_metadata %139 : memref<4x33x16x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %126
      scf.if %30 {
        %reinterpret_cast_167 = memref.reinterpret_cast %base_buffer_82 to offset: [0], sizes: [4, 512], strides: [528, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
        %base_buffer_168, %offset_169, %sizes_170:4, %strides_171:4 = memref.extract_strided_metadata %135 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> -> memref<bf16, #hivm.address_space<cbuf>>, index, index, index, index, index, index, index, index, index
        %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_168 to offset: [0], sizes: [4, 512], strides: [512, 1] : memref<bf16, #hivm.address_space<cbuf>> to memref<4x512xbf16, #hivm.address_space<cbuf>>
        %cast_173 = memref.cast %reinterpret_cast_167 : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_174 = memref.cast %reinterpret_cast_172 : memref<4x512xbf16, #hivm.address_space<cbuf>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @copy_ubuf_to_cbuf_2d_bfloat16_t(%cast_173, %cast_174) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      } {limit_sub_block_id0}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      %140 = arith.addi %96, %c2_i64 : i64
      memref.store %140, %alloca_1[%c0] : memref<1xi64>
      %141 = arith.addi %96, %c3_i64 : i64
      memref.store %141, %alloca_1[%c0] : memref<1xi64>
      %142 = arith.addi %96, %c4_i64 : i64
      memref.store %142, %alloca_1[%c0] : memref<1xi64>
      %143 = hivm.hir.pointer_cast(%c238848_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_10(%138, %42, %143) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_11(%38, %arg14, %38) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, f32, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_12(%38, %43, %42, %38, %43) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_13(%35, %43, %43) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      %base_buffer_86, %offset_87, %sizes_88, %strides_89 = memref.extract_strided_metadata %arg2 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_90 = memref.reinterpret_cast %base_buffer_86 to offset: [%71], sizes: [%81, %83], strides: [256, 1] : memref<bf16, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_91, %offset_92, %sizes_93:2, %strides_94:2 = memref.extract_strided_metadata %41 : memref<64x64xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_95 = memref.reinterpret_cast %base_buffer_91 to offset: [0], sizes: [%81, %83], strides: [64, 1] : memref<bf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %86 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_6(%41) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_96 = memref.cast %reinterpret_cast_90 : memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_97 = memref.cast %reinterpret_cast_95 : memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_bfloat16_t(%cast_96, %cast_97, %c2_i32, %cst_0, %c0, %c0_i32) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
      %base_buffer_98, %offset_99, %sizes_100, %strides_101 = memref.extract_strided_metadata %arg3 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_102 = memref.reinterpret_cast %base_buffer_98 to offset: [%71], sizes: [%81, %83], strides: [256, 1] : memref<bf16, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_103, %offset_104, %sizes_105:2, %strides_106:2 = memref.extract_strided_metadata %40 : memref<64x64xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_107 = memref.reinterpret_cast %base_buffer_103 to offset: [0], sizes: [%81, %83], strides: [64, 1] : memref<bf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %86 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_7(%40) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_108 = memref.cast %reinterpret_cast_102 : memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_109 = memref.cast %reinterpret_cast_107 : memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_bfloat16_t(%cast_108, %cast_109, %c2_i32, %cst_0, %c0, %c0_i32) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %46]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_14(%40, %35) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      %144 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_15(%43, %35, %144) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      %base_buffer_110, %offset_111, %sizes_112, %strides_113 = memref.extract_strided_metadata %arg9 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_114 = memref.reinterpret_cast %base_buffer_110 to offset: [%71], sizes: [%81, %83], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_115, %offset_116, %sizes_117:2, %strides_118:2 = memref.extract_strided_metadata %39 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_119 = memref.reinterpret_cast %base_buffer_115 to offset: [0], sizes: [%81, %83], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %86 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_8(%39) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_120 = memref.cast %reinterpret_cast_114 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_121 = memref.cast %reinterpret_cast_119 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_120, %cast_121, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_16(%38, %39, %38) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
      %base_buffer_122, %offset_123, %sizes_124, %strides_125 = memref.extract_strided_metadata %arg10 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_126 = memref.reinterpret_cast %base_buffer_122 to offset: [%71], sizes: [%81, %83], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_127, %offset_128, %sizes_129:2, %strides_130:2 = memref.extract_strided_metadata %37 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_131 = memref.reinterpret_cast %base_buffer_127 to offset: [0], sizes: [%81, %83], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %86 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_9(%37) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_132 = memref.cast %reinterpret_cast_126 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_133 = memref.cast %reinterpret_cast_131 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_132, %cast_133, %c2_i32, %cst, %c0, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_17(%43, %37, %36) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_18(%41, %38, %35, %36, %35) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
      %cast_134 = memref.cast %35 : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_135 = memref.cast %43 : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_136 = memref.cast %43 : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @cumsum_2d_float_dim0_comp(%cast_134, %cast_135, %cast_136, %false) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i1) -> ()
      %145 = hivm.hir.pointer_cast(%c239104_i64) : memref<64xf32, #hivm.address_space<ub>>
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_19(%35, %145) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_20(%35, %43, %145, %143, %144, %35) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %45]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
      %base_buffer_137, %offset_138, %sizes_139:2, %strides_140:2 = memref.extract_strided_metadata %38 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_141 = memref.reinterpret_cast %base_buffer_137 to offset: [0], sizes: [%81, %83], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %base_buffer_142, %offset_143, %sizes_144, %strides_145 = memref.extract_strided_metadata %arg11 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_146 = memref.reinterpret_cast %base_buffer_142 to offset: [%71], sizes: [%81, %83], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
      scf.if %30 {
        %cast_167 = memref.cast %reinterpret_cast_141 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_168 = memref.cast %reinterpret_cast_146 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_167, %cast_168, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {limit_sub_block_id0}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 6
      %base_buffer_147, %offset_148, %sizes_149:2, %strides_150:2 = memref.extract_strided_metadata %36 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_151 = memref.reinterpret_cast %base_buffer_147 to offset: [0], sizes: [%81, %83], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %base_buffer_152, %offset_153, %sizes_154, %strides_155 = memref.extract_strided_metadata %arg12 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_156 = memref.reinterpret_cast %base_buffer_152 to offset: [%71], sizes: [%81, %83], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      scf.if %30 {
        %cast_167 = memref.cast %reinterpret_cast_151 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_168 = memref.cast %reinterpret_cast_156 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_167, %cast_168, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {limit_sub_block_id0}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      %base_buffer_157, %offset_158, %sizes_159:2, %strides_160:2 = memref.extract_strided_metadata %35 : memref<64x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_161 = memref.reinterpret_cast %base_buffer_157 to offset: [0], sizes: [%81, %83], strides: [64, 1] : memref<f32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %base_buffer_162, %offset_163, %sizes_164, %strides_165 = memref.extract_strided_metadata %arg13 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_166 = memref.reinterpret_cast %base_buffer_162 to offset: [%71], sizes: [%81, %83], strides: [256, 1] : memref<f32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %30 {
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_167 = memref.cast %reinterpret_cast_161 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_168 = memref.cast %reinterpret_cast_166 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_167, %cast_168, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {limit_sub_block_id0}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %46]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 8
      hivm.hir.set_ctrl true at ctrl[60]
      %146 = arith.addi %31, %c1_i64 : i64
      memref.store %146, %alloca[%c0] : memref<1xi64>
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 5
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
```

在 mcf-off 版本中，相关 SSA 与 mcf-on 的对照关系如下：

| mcf-on SSA | mcf-off SSA | mcf-off 地址复用 |
| --- | --- | --- |
| %108 | %109 | %2 前半部分 |
| %106 | %107 | %3 前半部分 |
| %130 | %131 | %2 后半部分 |
| %128 | %129 | %3 后半部分 |

其中 mcf-off 中 `%109, %107, %131, %129` 与 `%36` 存在地址复用

```mlir
%2 = hivm.hir.pointer_cast(%c33280_i64) : memref<64x64xf32, #hivm.address_space<ub>>
%3 = hivm.hir.pointer_cast(%c49664_i64) : memref<64x64xf32, #hivm.address_space<ub>>
%36 = arith.select %34, %3, %2 : memref<64x64xf32, #hivm.address_space<ub>>
```

```text

UB 地址          33280          41472          49664          57856          66048
                  |--------------|--------------|--------------|--------------|
H/DH 阶段        |  H：%109      |  H：%131      |  DH：%107     |  DH：%129     |
                  |-----------------------------|-----------------------------|
DK 输出阶段      |       %2：16 KiB             |       %3：16 KiB             |
                  \_____________________________ _____________________________/
                                                V
                                  %36 按外层迭代奇偶选择其中一个槽
```

这里与 mcf-on 的区别在于 `%36` 的生命周期更短，它在较早的时候就已经 store 到 gm

> `%36` 对应于 mcf-on 版本中的 `%36` ，在 mcf-on 中尝试对 `%35` 的复用修改为对 `%36` 的复用调整 barrier 之后能够恢复到 mcf-off 的性能

## 附件

### ttadapter

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
  func.func private @triton_cumsum(tensor<64x64xf32>, i32, i1) -> tensor<64x64xf32>
  func.func @chunk_gla_bwd_kernel_inter(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg12: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg13: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg14: f32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c2_i32 = arith.constant 2 : i32
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %c32 = arith.constant 32 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c4_i64 = arith.constant 4 : i64
    %c64_i32 = arith.constant 64 : i32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = tensor.empty() : tensor<64xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %4 = arith.extsi %arg20 : i32 to i64
    %5 = arith.extsi %arg21 : i32 to i64
    %6 = arith.divsi %5, %c4_i64 : i64
    %7 = arith.remsi %5, %c4_i64 : i64
    %8 = arith.extsi %arg15 : i32 to i64
    %9 = arith.addi %arg15, %c63_i32 : i32
    %10 = arith.divsi %9, %c64_i32 : i32
    %11 = arith.extsi %10 : i32 to i64
    %12 = arith.muli %6, %11 : i64
    %13 = arith.addi %12, %4 : i64
    %14 = arith.muli %6, %8 : i64
    %15 = arith.muli %arg19, %c64_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.muli %14, %c4_i64 : i64
    %18 = arith.addi %17, %7 : i64
    %19 = arith.muli %18, %c64_i64 : i64
    %20 = arith.index_cast %19 : i64 to index
    %21 = arith.muli %13, %c4_i64 : i64
    %22 = arith.addi %21, %7 : i64
    %23 = arith.muli %22, %c4096_i64 : i64
    %24 = arith.index_cast %23 : i64 to index
    %25 = arith.muli %4, %c64_i64 : i64
    %26 = arith.index_cast %25 : i64 to index
    %27 = arith.muli %26, %c256 : index
    %28 = arith.addi %20, %27 : index
    %29 = arith.addi %28, %16 : index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [64, 64], strides: [256, 1] : memref<?xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x64xf32>
    %30 = arith.addi %26, %c64 : index
    %31 = arith.index_cast %arg15 : i32 to index
    %32 = arith.maxsi %26, %31 : index
    %33 = arith.minsi %30, %32 : index
    %34 = arith.subi %33, %26 : index
    %35 = arith.addi %16, %c64 : index
    %36 = arith.maxsi %16, %c64 : index
    %37 = arith.minsi %35, %36 : index
    %38 = arith.subi %37, %16 : index
    %39 = arith.minsi %34, %c64 : index
    %40 = arith.maxsi %39, %c0 : index
    %41 = arith.minsi %38, %c64 : index
    %42 = arith.maxsi %41, %c0 : index
    %43 = arith.cmpi slt, %40, %c64 : index
    %44 = arith.cmpi slt, %42, %c64 : index
    %45 = arith.ori %43, %44 : i1
    scf.if %45 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<64x64xf32>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    %subview_1 = memref.subview %alloc[0, 0] [%40, %42] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    memref.copy %subview, %subview_1 : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
    %46 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %47 = arith.addi %25, %c64_i64 : i64
    %48 = arith.minsi %8, %47 : i64
    %49 = arith.subi %48, %c1_i64 : i64
    %50 = arith.muli %49, %c256_i64 : i64
    %51 = arith.addi %19, %50 : i64
    %52 = arith.index_cast %51 : i64 to index
    %53 = arith.addi %52, %16 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%53], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<64xf32>
    %54 = arith.cmpi slt, %38, %c64 : index
    scf.if %54 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc_3 : memref<64xf32>)
    } {hivm.unlikely_condition}
    %subview_4 = memref.subview %reinterpret_cast_2[0] [%38] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_5 = memref.subview %alloc_3[0] [%38] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_4, %subview_5 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %55 = bufferization.to_tensor %alloc_3 restrict writable : memref<64xf32>
    %56:3 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %1, %arg24 = %1, %arg25 = %3) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64xf32>)  : i32 {
      %83 = arith.muli %arg22, %c32_i32 : i32
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.addi %28, %84 : index
      %reinterpret_cast_33 = memref.reinterpret_cast %arg4 to offset: [%85], sizes: [64, 32], strides: [256, 1] : memref<?xbf16> to memref<64x32xbf16, strided<[256, 1], offset: ?>>
      %reinterpret_cast_34 = memref.reinterpret_cast %arg7 to offset: [%85], sizes: [64, 32], strides: [256, 1] : memref<?xbf16> to memref<64x32xbf16, strided<[256, 1], offset: ?>>
      %alloc_35 = memref.alloc() : memref<64x32xbf16>
      %86 = arith.addi %84, %c32 : index
      %87 = arith.maxsi %84, %c64 : index
      %88 = arith.minsi %86, %87 : index
      %89 = arith.subi %88, %84 : index
      %90 = arith.minsi %89, %c32 : index
      %91 = arith.maxsi %90, %c0 : index
      %92 = arith.cmpi slt, %91, %c32 : index
      %93 = arith.ori %43, %92 : i1
      scf.if %93 {
        linalg.fill ins(%cst : bf16) outs(%alloc_35 : memref<64x32xbf16>)
      } {hivm.unlikely_condition}
      %subview_36 = memref.subview %reinterpret_cast_33[0, 0] [%40, %91] [1, 1] : memref<64x32xbf16, strided<[256, 1], offset: ?>> to memref<?x?xbf16, strided<[256, 1], offset: ?>>
      %subview_37 = memref.subview %alloc_35[0, 0] [%40, %91] [1, 1] : memref<64x32xbf16> to memref<?x?xbf16, strided<[32, 1]>>
      memref.copy %subview_36, %subview_37 : memref<?x?xbf16, strided<[256, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1]>>
      %94 = bufferization.to_tensor %alloc_35 restrict writable : memref<64x32xbf16>
      %alloc_38 = memref.alloc() : memref<64x32xbf16>
      scf.if %93 {
        linalg.fill ins(%cst : bf16) outs(%alloc_38 : memref<64x32xbf16>)
      } {hivm.unlikely_condition}
      %subview_39 = memref.subview %reinterpret_cast_34[0, 0] [%40, %91] [1, 1] : memref<64x32xbf16, strided<[256, 1], offset: ?>> to memref<?x?xbf16, strided<[256, 1], offset: ?>>
      %subview_40 = memref.subview %alloc_38[0, 0] [%40, %91] [1, 1] : memref<64x32xbf16> to memref<?x?xbf16, strided<[32, 1]>>
      memref.copy %subview_39, %subview_40 : memref<?x?xbf16, strided<[256, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1]>>
      %95 = bufferization.to_tensor %alloc_38 restrict writable : memref<64x32xbf16>
      %96 = arith.addi %24, %84 : index
      %97 = arith.muli %16, %c64 : index
      %98 = arith.addi %96, %97 : index
      %reinterpret_cast_41 = memref.reinterpret_cast %arg6 to offset: [%98], sizes: [64, 32], strides: [64, 1] : memref<?xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %alloc_42 = memref.alloc() : memref<64x32xf32>
      %99 = arith.ori %44, %92 : i1
      scf.if %99 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc_42 : memref<64x32xf32>)
      } {hivm.unlikely_condition}
      %subview_43 = memref.subview %reinterpret_cast_41[0, 0] [%42, %91] [1, 1] : memref<64x32xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %subview_44 = memref.subview %alloc_42[0, 0] [%42, %91] [1, 1] : memref<64x32xf32> to memref<?x?xf32, strided<[32, 1]>>
      memref.copy %subview_43, %subview_44 : memref<?x?xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1]>>
      %100 = bufferization.to_tensor %alloc_42 restrict writable : memref<64x32xf32>
      %101 = tensor.empty() : tensor<32x64xf32>
      %transposed = linalg.transpose ins(%100 : tensor<64x32xf32>) outs(%101 : tensor<32x64xf32>) permutation = [1, 0]
      %reinterpret_cast_45 = memref.reinterpret_cast %arg8 to offset: [%98], sizes: [64, 32], strides: [64, 1] : memref<?xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
      %alloc_46 = memref.alloc() : memref<64x32xf32>
      scf.if %99 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc_46 : memref<64x32xf32>)
      } {hivm.unlikely_condition}
      %subview_47 = memref.subview %reinterpret_cast_45[0, 0] [%42, %91] [1, 1] : memref<64x32xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %subview_48 = memref.subview %alloc_46[0, 0] [%42, %91] [1, 1] : memref<64x32xf32> to memref<?x?xf32, strided<[32, 1]>>
      memref.copy %subview_47, %subview_48 : memref<?x?xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1]>>
      %102 = bufferization.to_tensor %alloc_46 restrict writable : memref<64x32xf32>
      %transposed_49 = linalg.transpose ins(%102 : tensor<64x32xf32>) outs(%101 : tensor<32x64xf32>) permutation = [1, 0]
      %103 = arith.mulf %transposed, %transposed_49 : tensor<32x64xf32>
      %reduced_50 = linalg.reduce ins(%103 : tensor<32x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %109 = arith.addf %in, %init : f32
          linalg.yield %109 : f32
        }
      %104 = arith.addf %arg25, %reduced_50 : tensor<64xf32>
      %105 = arith.truncf %transposed : tensor<32x64xf32> to tensor<32x64xbf16>
      %106 = linalg.matmul {indexing_maps = [#map, #map1, #map2], input_precision = "ieee"} ins(%95, %105 : tensor<64x32xbf16>, tensor<32x64xbf16>) outs(%arg23 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %107 = arith.truncf %transposed_49 : tensor<32x64xf32> to tensor<32x64xbf16>
      %108 = linalg.matmul {indexing_maps = [#map, #map1, #map2], input_precision = "ieee"} ins(%94, %107 : tensor<64x32xbf16>, tensor<32x64xbf16>) outs(%arg24 : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %106, %108, %104 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64xf32>
    }
    %57 = math.exp2 %55 : tensor<64xf32>
    %58 = arith.mulf %56#2, %57 : tensor<64xf32>
    %59 = linalg.fill ins(%arg14 : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %60 = arith.mulf %56#0, %59 : tensor<64x64xf32>
    %61 = math.exp2 %46 : tensor<64x64xf32>
    %62 = arith.mulf %60, %61 : tensor<64x64xf32>
    %broadcasted = linalg.broadcast ins(%55 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [0]
    %63 = arith.subf %broadcasted, %46 : tensor<64x64xf32>
    %64 = math.exp2 %63 : tensor<64x64xf32>
    %65 = arith.mulf %56#1, %64 : tensor<64x64xf32>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [64, 64], strides: [256, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[256, 1], offset: ?>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [64, 64], strides: [256, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[256, 1], offset: ?>>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg9 to offset: [%29], sizes: [64, 64], strides: [256, 1] : memref<?xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg10 to offset: [%29], sizes: [64, 64], strides: [256, 1] : memref<?xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
    %alloc_10 = memref.alloc() : memref<64x64xbf16>
    scf.if %45 {
      linalg.fill ins(%cst : bf16) outs(%alloc_10 : memref<64x64xbf16>)
    } {hivm.unlikely_condition}
    %subview_11 = memref.subview %reinterpret_cast_6[0, 0] [%40, %42] [1, 1] : memref<64x64xbf16, strided<[256, 1], offset: ?>> to memref<?x?xbf16, strided<[256, 1], offset: ?>>
    %subview_12 = memref.subview %alloc_10[0, 0] [%40, %42] [1, 1] : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1]>>
    memref.copy %subview_11, %subview_12 : memref<?x?xbf16, strided<[256, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1]>>
    %66 = bufferization.to_tensor %alloc_10 restrict writable : memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    scf.if %45 {
      linalg.fill ins(%cst : bf16) outs(%alloc_13 : memref<64x64xbf16>)
    } {hivm.unlikely_condition}
    %subview_14 = memref.subview %reinterpret_cast_7[0, 0] [%40, %42] [1, 1] : memref<64x64xbf16, strided<[256, 1], offset: ?>> to memref<?x?xbf16, strided<[256, 1], offset: ?>>
    %subview_15 = memref.subview %alloc_13[0, 0] [%40, %42] [1, 1] : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1]>>
    memref.copy %subview_14, %subview_15 : memref<?x?xbf16, strided<[256, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1]>>
    %67 = bufferization.to_tensor %alloc_13 restrict writable : memref<64x64xbf16>
    %68 = arith.extf %67 : tensor<64x64xbf16> to tensor<64x64xf32>
    %69 = arith.mulf %65, %68 : tensor<64x64xf32>
    %reduced = linalg.reduce ins(%69 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %83 = arith.addf %in, %init : f32
        linalg.yield %83 : f32
      }
    %70 = arith.addf %58, %reduced : tensor<64xf32>
    %alloc_16 = memref.alloc() : memref<64x64xf32>
    scf.if %45 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc_16 : memref<64x64xf32>)
    } {hivm.unlikely_condition}
    %subview_17 = memref.subview %reinterpret_cast_8[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    %subview_18 = memref.subview %alloc_16[0, 0] [%40, %42] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    memref.copy %subview_17, %subview_18 : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
    %71 = bufferization.to_tensor %alloc_16 restrict writable : memref<64x64xf32>
    %72 = arith.addf %62, %71 : tensor<64x64xf32>
    %alloc_19 = memref.alloc() : memref<64x64xf32>
    scf.if %45 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc_19 : memref<64x64xf32>)
    } {hivm.unlikely_condition}
    %subview_20 = memref.subview %reinterpret_cast_9[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    %subview_21 = memref.subview %alloc_19[0, 0] [%40, %42] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    memref.copy %subview_20, %subview_21 : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
    %73 = bufferization.to_tensor %alloc_19 restrict writable : memref<64x64xf32>
    %74 = arith.addf %65, %73 : tensor<64x64xf32>
    %75 = arith.extf %66 : tensor<64x64xbf16> to tensor<64x64xf32>
    %76 = arith.mulf %75, %72 : tensor<64x64xf32>
    %77 = arith.mulf %68, %74 : tensor<64x64xf32>
    %78 = arith.subf %76, %77 : tensor<64x64xf32>
    %79 = call @triton_cumsum(%78, %c0_i32, %false) : (tensor<64x64xf32>, i32, i1) -> tensor<64x64xf32>
    %80 = arith.subf %78, %79 : tensor<64x64xf32>
    %reduced_22 = linalg.reduce ins(%78 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %83 = arith.addf %in, %init : f32
        linalg.yield %83 : f32
      }
    %broadcasted_23 = linalg.broadcast ins(%reduced_22 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [0]
    %81 = arith.addf %80, %broadcasted_23 : tensor<64x64xf32>
    %broadcasted_24 = linalg.broadcast ins(%70 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [0]
    %82 = arith.addf %81, %broadcasted_24 : tensor<64x64xf32>
    %reinterpret_cast_25 = memref.reinterpret_cast %arg11 to offset: [%29], sizes: [64, 64], strides: [256, 1] : memref<?xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
    %reinterpret_cast_26 = memref.reinterpret_cast %arg12 to offset: [%29], sizes: [64, 64], strides: [256, 1] : memref<?xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
    %reinterpret_cast_27 = memref.reinterpret_cast %arg13 to offset: [%29], sizes: [64, 64], strides: [256, 1] : memref<?xf32> to memref<64x64xf32, strided<[256, 1], offset: ?>>
    %extracted_slice = tensor.extract_slice %72[0, 0] [%40, %42] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    %subview_28 = memref.subview %reinterpret_cast_25[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_28 : (tensor<?x?xf32>, memref<?x?xf32, strided<[256, 1], offset: ?>>) -> ()
    %extracted_slice_29 = tensor.extract_slice %74[0, 0] [%40, %42] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    %subview_30 = memref.subview %reinterpret_cast_26[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_29 in writable %subview_30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[256, 1], offset: ?>>) -> ()
    %extracted_slice_31 = tensor.extract_slice %82[0, 0] [%40, %42] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    %subview_32 = memref.subview %reinterpret_cast_27[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_31 in writable %subview_32 : (tensor<?x?xf32>, memref<?x?xf32, strided<[256, 1], offset: ?>>) -> ()
    return
  }
}
```

编译命令：

```sh
bishengir-compile \
  --target=Ascend950PR_9579 \
  --enable-auto-multi-buffer=True \
  --enable-auto-bind-sub-block=True \
  --enable-optimized-metaop=False \
  --disable-ffts \
  --limit-auto-multi-buffer-of-local-buffer=no-limit \
  --enable-auto-blockify-loop \
  --enable-hfusion-compile=true \
  --enable-triton-kernel-compile=true \
  --enable-vf-merge-level=1 \
  --hfusion-enable-multiple-consumer-fusion=true
```

## GSS 同步等待分析

在将对 `%35` 的复用修改为对 `%36` 的复用之后，GSS 还有需要修复的地方：

GSS 生成的 mlir 如下

```mlir
func.func @chunk_gla_bwd_kernel_inter_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg13: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg14: f32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<19xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c4_i32 = arith.constant 4 : i32
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c2_i64 = arith.constant 2 : i64
    %c239104_i64 = arith.constant 239104 : i64
    %c213760_i64 = arith.constant 213760 : i64
    %c230144_i64 = arith.constant 230144 : i64
    %c221952_i64 = arith.constant 221952 : i64
    %c205568_i64 = arith.constant 205568 : i64
    %c189184_i64 = arith.constant 189184 : i64
    %c172800_i64 = arith.constant 172800 : i64
    %c131840_i64 = arith.constant 131840 : i64
    %c123648_i64 = arith.constant 123648 : i64
    %c115456_i64 = arith.constant 115456 : i64
    %c107264_i64 = arith.constant 107264 : i64
    %c238848_i64 = arith.constant 238848 : i64
    %c156416_i64 = arith.constant 156416 : i64
    %c140032_i64 = arith.constant 140032 : i64
    %c90880_i64 = arith.constant 90880 : i64
    %c74496_i64 = arith.constant 74496 : i64
    %c70272_i64 = arith.constant 70272 : i64
    %c66048_i64 = arith.constant 66048 : i64
    %c49664_i64 = arith.constant 49664 : i64
    %c33280_i64 = arith.constant 33280 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c238592_i64 = arith.constant 238592 : i64
    %c33024_i64 = arith.constant 33024 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c0_i64 = arith.constant 0 : i64
    %c238336_i64 = arith.constant 238336 : i64
    %c28_i32 = arith.constant 28 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %c32 = arith.constant 32 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c4_i64 = arith.constant 4 : i64
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.muli %arg16, %arg17 : i32
    %1 = arith.muli %0, %arg18 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.pointer_cast(%c238336_i64) : memref<64xf32, #hivm.address_space<ub>>
    call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_20(%4) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
    %5 = arith.extsi %arg15 : i32 to i64
    %6 = arith.addi %arg15, %c63_i32 : i32
    %7 = arith.divsi %6, %c64_i32 : i32
    %8 = arith.extsi %7 : i32 to i64
    %9 = arith.index_cast %arg15 : i32 to index
    %10 = hivm.hir.get_sub_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = arith.cmpi eq, %11, %c0 : index
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 7
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 7
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 9
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 9
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg19 = %3 to %1 step %c28_i32  : i32 {
      %13 = hivm.hir.multi_buffer_counter -> i64
      %c2_i64_1 = arith.constant 2 : i64
      %14 = arith.remui %13, %c2_i64_1 : i64
      %15 = arith.index_cast %14 : i64 to index
      %16 = arith.index_cast %15 : index to i1
      %c4_i64_2 = arith.constant 4 : i64
      %c5_i64 = arith.constant 5 : i64
      %17 = arith.select %16, %c4_i64_2, %c5_i64 : i64
      %18 = hivm.hir.pointer_cast(%c33280_i64, %c49664_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %19 = hivm.hir.pointer_cast(%c205568_i64, %c221952_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %20 = hivm.hir.pointer_cast(%c172800_i64, %c189184_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %21 = hivm.hir.pointer_cast(%c74496_i64, %c90880_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %22 = hivm.hir.pointer_cast(%c140032_i64, %c156416_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %23 = hivm.hir.pointer_cast(%c123648_i64, %c131840_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
      annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<64x64xbf16, #hivm.address_space<ub>>
      %24 = hivm.hir.pointer_cast(%c107264_i64, %c115456_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
      annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<64x64xbf16, #hivm.address_space<ub>>
      %25 = hivm.hir.pointer_cast(%c32768_i64, %c33024_i64) : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
      %26 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %27 = arith.remsi %arg19, %arg16 : i32
      %28 = arith.divsi %arg19, %arg16 : i32
      %29 = arith.remsi %28, %arg17 : i32
      %30 = arith.divsi %arg19, %0 : i32
      %31 = arith.remsi %30, %arg18 : i32
      %32 = arith.extsi %29 : i32 to i64
      %33 = arith.extsi %31 : i32 to i64
      %34 = arith.divsi %33, %c4_i64 : i64
      %35 = arith.remsi %33, %c4_i64 : i64
      %36 = arith.muli %34, %8 : i64
      %37 = arith.addi %36, %32 : i64
      %38 = arith.muli %34, %5 : i64
      %39 = arith.muli %27, %c64_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.muli %38, %c4_i64 : i64
      %42 = arith.addi %41, %35 : i64
      %43 = arith.muli %42, %c64_i64 : i64
      %44 = arith.index_cast %43 : i64 to index
      %45 = arith.muli %37, %c4_i64 : i64
      %46 = arith.addi %45, %35 : i64
      %47 = arith.muli %46, %c4096_i64 : i64
      %48 = arith.index_cast %47 : i64 to index
      %49 = arith.muli %32, %c64_i64 : i64
      %50 = arith.index_cast %49 : i64 to index
      %51 = affine.apply #map2()[%40, %44, %50]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [64, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %52 = affine.apply #map()[%50]
      %53 = arith.maxsi %50, %9 : index
      %54 = arith.minsi %52, %53 : index
      %55 = affine.apply #map1()[%54, %50]
      %56 = affine.apply #map()[%40]
      %57 = arith.maxsi %40, %c64 : index
      %58 = arith.minsi %56, %57 : index
      %59 = affine.apply #map1()[%58, %40]
      %60 = arith.minsi %55, %c64 : index
      %61 = arith.maxsi %60, %c0 : index
      %62 = arith.minsi %59, %c64 : index
      %63 = arith.maxsi %62, %c0 : index
      %64 = arith.cmpi slt, %61, %c64 : index
      %65 = arith.cmpi slt, %63, %c64 : index
      %66 = arith.ori %64, %65 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %26[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %66 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_0(%26) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %17]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32
      %67 = arith.addi %49, %c64_i64 : i64
      %68 = arith.minsi %5, %67 : i64
      %69 = arith.subi %68, %c1_i64 : i64
      %70 = arith.muli %69, %c256_i64 : i64
      %71 = arith.addi %43, %70 : i64
      %72 = arith.index_cast %71 : i64 to index
      %73 = affine.apply #map7()[%72, %40]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%73], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %74 = arith.cmpi slt, %59, %c64 : index
      %subview_5 = memref.subview %reinterpret_cast_4[0] [%59] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %25[0] [%59] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %74 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_1(%25) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.load ins(%subview_5 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %alloca = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
      memref.store %c0_i32, %alloca[] : memref<i32>
      %alloca_7 = memref.alloca() {normalize_matmul_counter = 1 : i32} : memref<i32>
      memref.store %c0_i32, %alloca_7[] : memref<i32>
      %75 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%4 : memref<64xf32, #hivm.address_space<ub>>) outs(%75 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      %76 = scf.for %arg20 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg21 = %75) -> (memref<64xf32, #hivm.address_space<ub>>)  : i32 {
        %81 = arith.subi %arg20, %c1_i32 : i32
        %82 = hivm.hir.multi_buffer_counter -> i64
        %c2_i64_29 = arith.constant 2 : i64
        %83 = arith.remui %82, %c2_i64_29 : i64
        %84 = arith.index_cast %83 : i64 to index
        %85 = arith.index_cast %84 : index to i1
        %c2_i64_30 = arith.constant 2 : i64
        %c3_i64 = arith.constant 3 : i64
        %86 = arith.select %85, %c2_i64_30, %c3_i64 : i64
        %87 = hivm.hir.multi_buffer_counter -> i64
        %88 = arith.remui %87, %c2_i64 : i64
        %89 = arith.index_cast %88 : i64 to index
        %90 = arith.addi %89, %c3 : index
        %91 = arith.index_cast %90 : index to i64
        %92 = arith.addi %89, %c5 : index
        %93 = arith.index_cast %92 : index to i64
        %94 = hivm.hir.pointer_cast(%c221952_i64, %c230144_i64) : memref<32x64xf32, #hivm.address_space<ub>>
        %95 = hivm.hir.pointer_cast(%c221952_i64, %c230144_i64) : memref<32x64xf32, #hivm.address_space<ub>>
        annotation.mark %94 {hivm.multi_buffer = 2 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
        annotation.mark %95 {hivm.multi_buffer = 2 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
        %96 = hivm.hir.pointer_cast(%c205568_i64, %c213760_i64) : memref<32x64xf32, #hivm.address_space<ub>>
        %97 = hivm.hir.pointer_cast(%c205568_i64, %c213760_i64) : memref<32x64xf32, #hivm.address_space<ub>>
        annotation.mark %96 {hivm.multi_buffer = 2 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
        annotation.mark %97 {hivm.multi_buffer = 2 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
        %98 = hivm.hir.pointer_cast(%c4096_i64, %c20480_i64) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        %99 = hivm.hir.pointer_cast(%c20480_i64, %c4096_i64) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %99 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %98 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        %100 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        %101 = hivm.hir.pointer_cast(%c16384_i64, %c0_i64) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %101 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %100 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        %102 = arith.cmpi sge, %arg20, %c0_i32 : i32
        %103 = arith.cmpi slt, %arg20, %c2_i32 : i32
        %104 = arith.andi %102, %103 : i1
        annotation.mark %104 {hivm.preload_num = 1 : i32} : i1
        %105 = scf.if %104 -> (memref<64xf32, #hivm.address_space<ub>>) {
          %109 = arith.muli %arg20, %c32_i32 : i32
          %110 = arith.index_cast %109 : i32 to index
          %111 = affine.apply #map3()[%110]
          %112 = arith.maxsi %110, %c64 : index
          %113 = arith.minsi %111, %112 : index
          %114 = affine.apply #map1()[%113, %110]
          %115 = arith.minsi %114, %c32 : index
          %116 = arith.maxsi %115, %c0 : index
          %117 = arith.cmpi slt, %116, %c32 : index
          %118 = affine.apply #map8()[%48, %110, %40]
          %119 = arith.ori %65, %117 : i1
          %reinterpret_cast_31 = memref.reinterpret_cast %arg6 to offset: [%118], sizes: [32, 64], strides: [1, 64] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
          %subview_32 = memref.subview %reinterpret_cast_31[0, 0] [%116, %63] [1, 1] : memref<32x64xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
          scf.if %119 {
            func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_2(%97) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %subview_33 = memref.subview %97[0, 0] [%116, 64] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %86]
          hivm.hir.load ins(%subview_32 : memref<?x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>) outs(%subview_33 : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32
          %reinterpret_cast_34 = memref.reinterpret_cast %arg8 to offset: [%118], sizes: [32, 64], strides: [1, 64] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
          %subview_35 = memref.subview %reinterpret_cast_34[0, 0] [%116, %63] [1, 1] : memref<32x64xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
          scf.if %119 {
            func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_3(%95) {hivm.vector_function, no_inline} : (memref<32x64xf32, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %subview_36 = memref.subview %95[0, 0] [%116, 64] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
          hivm.hir.load ins(%subview_35 : memref<?x?xf32, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>) outs(%subview_36 : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
          %120 = hivm.hir.pointer_cast(%c66048_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
          %subview_37 = memref.subview %120[0, 0, 0, 0] [4, 32, 16, 1] [1, 1, 1, 1] : memref<4x33x16x1xbf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
          %121 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
          %122 = hivm.hir.pointer_cast(%c70272_i64) : memref<4x33x16x1xbf16, #hivm.address_space<ub>>
          %subview_38 = memref.subview %122[0, 0, 0, 0] [4, 32, 16, 1] [1, 1, 1, 1] : memref<4x33x16x1xbf16, #hivm.address_space<ub>> to memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
          func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_merged_vf_0(%97, %95, %subview_37, %arg21, %121, %95, %subview_38) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<32x64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>, memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %86]
          %expand_shape = memref.expand_shape %subview_37 [[0], [1, 2], [3]] output_shape [4, 2, 16, 16] : memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>> into memref<4x2x16x16xbf16, strided<[528, 256, 16, 1]>, #hivm.address_space<ub>>
          annotation.mark %98 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %91
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          scf.if %12 {
            %collapse_shape = memref.collapse_shape %expand_shape [[0], [1, 2, 3]] : memref<4x2x16x16xbf16, strided<[528, 256, 16, 1]>, #hivm.address_space<ub>> into memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
            %collapse_shape_40 = memref.collapse_shape %98 [[0], [1, 2, 3]] : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> into memref<4x512xbf16, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%collapse_shape : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_40 : memref<4x512xbf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
          } {limit_sub_block_id0}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
          %expand_shape_39 = memref.expand_shape %subview_38 [[0], [1, 2], [3]] output_shape [4, 2, 16, 16] : memref<4x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>> into memref<4x2x16x16xbf16, strided<[528, 256, 16, 1]>, #hivm.address_space<ub>>
          annotation.mark %100 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %93
          scf.if %12 {
            %collapse_shape = memref.collapse_shape %expand_shape_39 [[0], [1, 2, 3]] : memref<4x2x16x16xbf16, strided<[528, 256, 16, 1]>, #hivm.address_space<ub>> into memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>
            %collapse_shape_40 = memref.collapse_shape %100 [[0], [1, 2, 3]] : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> into memref<4x512xbf16, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%collapse_shape : memref<4x512xbf16, strided<[528, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_40 : memref<4x512xbf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
          } {limit_sub_block_id0}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          scf.yield %121 : memref<64xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %arg21 : memref<64xf32, #hivm.address_space<ub>>
        }
        %106 = arith.cmpi sge, %81, %c0_i32 : i32
        %107 = arith.cmpi slt, %81, %c2_i32 : i32
        %108 = arith.andi %106, %107 : i1
        annotation.mark %108 {hivm.preload_num = 0 : i32} : i1
        scf.if %108 {
          %109 = memref.load %alloca[] : memref<i32>
          %110 = arith.addi %109, %c1_i32 : i32
          memref.store %110, %alloca[] : memref<i32>
          %111 = memref.load %alloca_7[] : memref<i32>
          %112 = arith.addi %111, %c1_i32 : i32
          memref.store %112, %alloca_7[] : memref<i32>
        }
        scf.yield %105 : memref<64xf32, #hivm.address_space<ub>>
      } {hivm.max_preload_num = 2 : i32}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      annotation.mark %21 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %22 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x64xf32, #hivm.address_space<ub>>
      %77 = hivm.hir.pointer_cast(%c238848_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_10(%76, %25, %77) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_11(%21, %arg14, %21) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, f32, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_12(%21, %26, %25, %21, %26) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_13(%22, %26, %26) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%51], sizes: [64, 64], strides: [256, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x64xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [64, 64], strides: [256, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x64xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [%51], sizes: [64, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_11 = memref.reinterpret_cast %arg10 to offset: [%51], sizes: [64, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_12 = memref.subview %reinterpret_cast_8[0, 0] [%61, %63] [1, 1] : memref<64x64xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %24[0, 0] [%61, %63] [1, 1] : memref<64x64xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %66 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_6(%24) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_12 : memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16
      %subview_14 = memref.subview %reinterpret_cast_9[0, 0] [%61, %63] [1, 1] : memref<64x64xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_15 = memref.subview %23[0, 0] [%61, %63] [1, 1] : memref<64x64xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %66 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_7(%23) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_14 : memref<?x?xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_15 : memref<?x?xbf16, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %78 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_14(%23, %78) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      %79 = hivm.hir.pointer_cast(%c238592_i64) : memref<64xf32, #hivm.address_space<ub>>
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_15(%26, %78, %79) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      %subview_16 = memref.subview %reinterpret_cast_10[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_17 = memref.subview %22[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %66 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_8(%22) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%subview_16 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_17 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 9
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_16(%21, %22, %21) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
      %subview_18 = memref.subview %reinterpret_cast_11[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_19 = memref.subview %20[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %66 {
        func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_9(%20) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_18 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_19 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_17(%24, %21, %78, %26, %20, %20, %19) {hivm.vector_function, no_inline} : (memref<64x64xbf16, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %17]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
      hivm.hir.vcumsum {needs_compensation} ins(%20 : memref<64x64xf32, #hivm.address_space<ub>>) outs(%18 : memref<64x64xf32, #hivm.address_space<ub>>) cum_dims = [0] reverse = false
      %80 = hivm.hir.pointer_cast(%c239104_i64) : memref<64xf32, #hivm.address_space<ub>>
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_18(%20, %80) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      func.call @chunk_gla_bwd_kernel_inter_mix_aiv_outlined_vf_19(%20, %18, %80, %77, %79, %18) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_20 = memref.reinterpret_cast %arg11 to offset: [%51], sizes: [64, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_21 = memref.reinterpret_cast %arg12 to offset: [%51], sizes: [64, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_22 = memref.reinterpret_cast %arg13 to offset: [%51], sizes: [64, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_23 = memref.subview %21[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %subview_24 = memref.subview %reinterpret_cast_20[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
      scf.if %12 {
        hivm.hir.store ins(%subview_23 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_24 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
      } {limit_sub_block_id0}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 7
      %subview_25 = memref.subview %19[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %subview_26 = memref.subview %reinterpret_cast_21[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      scf.if %12 {
        hivm.hir.store ins(%subview_25 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_26 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
      } {limit_sub_block_id0}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      %subview_27 = memref.subview %18[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %subview_28 = memref.subview %reinterpret_cast_22[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %12 {
        hivm.hir.store ins(%subview_27 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_28 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
      } {limit_sub_block_id0}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 5
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 6
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
```

其中简化之后的伪代码如下：

```text
for outer_iteration in ...:
    # ... g/g_last 加载及其同步 ...

    copy_ub_to_ub(...)                       # 726

    WAIT(MTE3, MTE2, 0)                      # 727：等待上一外层的 DK 释放
    WAIT(MTE3, V,    0)                      # 728

    # 内层循环之前：初始化两个槽的许可
    SET(V, MTE2, 2)                          # 729
    SET(V, MTE2, 3)                          # 730

    for j in range(4):                      # 731：此时循环还没有展开
        counter = multi_buffer_counter()    # 733；本 case 等价于 4*i+j
        slot = counter % 2
        event = 3 if slot == 0 else 2        # 740：SSA %86

        # 注意：wait 和归还 set 都在这个条件里面
        if 0 <= j < 2:                      # 764–768
            # ... 边界填充及其同步 ...

            WAIT(V, MTE2, event)            # 788：取得当前槽的许可

            LOAD_H(H[slot])                 # 789
            LOAD_DH(DH[slot])               # 798
            SET(MTE2, V, 1)                 # 799

            # ... 等待上一批 UB→L1 搬运释放输出区 ...
            WAIT(MTE2, V, 1)                # 806：等待本次 H/DH 加载
            merged_vf_0(H[slot], DH[slot])   # 807

            SET(V, MTE3, 0)                 # 808：允许搬运 VF 输出
            SET(V, MTE2, event)             # 809：归还当前 H/DH 槽

            # ... V→MTE3 等待、UB→L1 搬运及跨核同步 ...

        if 0 <= j - 1 < 2:                  # 834–844
            update_matmul_counters()

    # 内层循环之后：消费最后归还的两个许可
    WAIT(V, MTE2, 2)                         # 848
    WAIT(V, MTE2, 3)                         # 849

    # ... 后续向量计算、DK 输出同步 ...
    STORE_DK(...)
    SET(MTE3, V,    0)                      # 940
    SET(MTE3, MTE2, 0)                      # 941：供下一外层等待
    # ... DG 输出等 ...
```

其中内层 `j` 循环的 double buffer 在循环入口插入了 `WAIT(V, MTE2, event)` 并且在循环末尾插入了 `SET(V, MTE2, event)` 而这里实际上 j = 0 和 j = 1 不会发生等待。后续 loop unroll 之后，两次迭代的 set/wait 被展开，就导致实际的同步等待变为了：

```text
循环前 SET(3) -> j=0 的 WAIT(3)
循环前 SET(2) -> j=1 的 WAIT(2)
```

导致 `merged_vf_0` 等待了 j loop 之前的时间初始化 set，也就是等到 `copy_ub_to_ub` 的完成

> 这里可以在编译器上分析迭代之间不可能存在同步等待关系，进而消除 set init/wait/set 的插入

## GSS 优化算法：优化 GSS 对跨迭代的冲突分析

```mlir
// 初始化两个反向事件
hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]

scf.for ... {
  // %event 是随双缓冲轮转的事件号
  scf.if %active {
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %event]
    hivm.hir.load ...

    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vadd ...

    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %event]
  }
}

hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]

```

对于两次迭代访问记录 `OccurrenceA` （后续简称 A）和 `OccurrenceB` （后续简称 B），我们可以理解为：

- A：第 t 轮迭代读写 ub
- B：第 t+d 轮迭代读写 ub
- d > 0

需要判断的是：B 是否会覆盖 A 尚未使用完成的内存。算法流程如下：

### 步骤一：确定需要检查的迭代距离 d 的集合

若 A 可能执行的迭代索引区间为 `[startA, end]` ，B 可能执行的迭代索引区间为 `[startB, endB]` ，由于对于跨迭代依赖只需要考虑 B 晚于 A 执行的情况，因此 `first = max(1, startB - startA)`，`last = endB - startA` （排除 `endB <= startA` 的情况）

> 例如：A 在执行轮次 [0, 1]，B 执行轮次 [0, 1]，则 [first, last] = [1, 1] 意味着只需要检查 *相隔一轮* 的访问是否会发生冲突

### 步骤二：检查物理内存区间重叠

假设 A 有 m 个缓冲 slot，B 有 n 个：

- A 的地址集合：`a[0], a[1], …, a[m - 1]`
- B 的地址集合：`b[0], b[1], …, b[n - 1]`

对于任意 slot index pair `(i, j)` 检查它们所代表的内存区间重叠情况，`A[i] = [a[i], a[i] + sizeA]`，`B[i] = [b[i], b[i] + sizeB]` 。对于一个 32B 的缓冲区来说：

| slot A | slot B | intervals | overlap |
| --- | --- | --- | --- |
| 0 | 0 | [0, 32), [0, 32) | true |
| 0 | 32 | [0, 32), [32, 64) | false |
| 0 | 16 | [0, 32), [16, 48) | true, overlap 16B |

### 步骤三：判断迭代间隔 d 是否可以被选中

记 A 执行时的迭代计数器为 `C` （忽略迭代回绕）则：

- A 选择的 slot index = C mod m
- B 选择的 slot index = (C + d) mod n

若 `a[i]` 和 `b[j]` 出现地址冲突，要让 A 选择 slot i 且 B 选择 slot j，则必须同时满足：

- C ≡ i (mod m)           =>      C ≡  i (mod m)
- C + d ≡ j (mod n)     =>      C ≡ j - d (mod n)

两个同余式能够兼容需要满足：`i ≡ j - d (mod gcd(m, n))` 即 `d ≡ j - i (mod gcd(m, n))`，其中 `gcd(m, n)` 表示 m 和 n 的最小公倍数。因此我们不再需要关注其实迭代索引 C

> 1. 对于通过 `scf.yield` 指令传递指针的情况保守不分析优化
> 2. 忽略 C 之后会变成保守判断，具体见下面的例子中 A[1] 和 B[1] 在迭代执行三次的时候不可能出现冲突，但当前算法会保守计算为冲突。属于复杂度和优化效果的平衡处理

举例说明：address A = [0, 32], address B = [0, 32], slot size = 32, m = n = 2, gcd(m, n) = 2。循环迭代执行三次，[first, last] = [1, 2]

| index A | index B | overlap | d | conflict |
| --- | --- | --- | --- | --- |
| 0, address = 0 | 0, address = 0 | true | d≡0 (mod 2) | true, d = 2 |
| 0, address = 0 | 1, address = 32 | false | / | false |
| 1, address = 32 | 0, address = 0 | false | / | false |
| 1, address = 32 | 1, address = 32 | true | d≡0 (mod 2) | true, d = 2 |

实际实现 conflict 判断时，可以按下面方式处理：

1. 计算 `g = gcd(m, n)` 和 `[first, last]`
2. 计算 `residue = (j - i) mod g`
3. 判断在区间 `[first, last]` 中是否存在 `d mod g == residue`  => `offset = (residue + g - first % g) % g;` , `return offset <= last - first;`

### 其它：边界情况处理

在遇到 `2^64` 截断回 0 的时候，实际会从 `C + d - 2^64` 变为 `d ≡ j - i + 2^64  (mod g)` 。因此代码中需要同时判断两种余数：

1. residue
2. (residue + 2^64 mod g) mod g

```text
function mayConflictAcrossIterations(A, B, rangeA, rangeB):
    [startA, endA] = rangeA
    [startB, endB] = rangeB

    minDistance = max(1, startB - endA)
    maxDistance = endB - startA

    if minDistance > maxDistance:
        return false

    modulus = gcd(A.slotCount, B.slotCount)

    for i in [0, A.slotCount):
        for j in [0, B.slotCount):
            if not intervalsOverlap(
                A.addresses[i], A.slotSize,
                B.addresses[j], B.slotSize):
                continue

            # A matching distance satisfies:
            # d ≡ j - i (mod modulus)
            residue = (j - i) mod modulus

            firstMatchingDistance = minDistance
                + (residue - minDistance) mod modulus

            if firstMatchingDistance <= maxDistance:
                return true

    return false
```
