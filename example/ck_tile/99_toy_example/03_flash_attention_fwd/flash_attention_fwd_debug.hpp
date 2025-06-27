// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

#include "../../../example/ck_tile/99_toy_example/02_gemm/block_gemm_pipeline_agmem_bgmem_creg.hpp"
#include "block_gemm_pipeline_problem.hpp"
#include "block_gemm_areg_bsmem_creg_v1.hpp"
#include "flash_attention_fwd_impl.hpp"

namespace ck_tile {

// S[M0, N0] = Q[M0, K0] * K[N0, K0]
// P[M0, N0] = Softmax(S[M0, N0])
// O[M0, N1] = P[M0, N0] * V[N1, N0]
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          typename ODataType,
          index_t kBlockSize,
          index_t kHeadDim,
          index_t kM0PerBlock,
          index_t kN0PerBlock,
          index_t kK0PerBlock,
          index_t kN1PerBlock,
          index_t kK1PerBlock>
struct FlashAttentionFwd
{
    __device__ void operator()(const QDataType* q_ptr,
                               const KDataType* k_ptr,
                               const VDataType* v_ptr,
                               ODataType* o_ptr,
                               const index_t M0,
                               const index_t N0,
                               const index_t K0,
                               const index_t N1,
                               const index_t /* Batch */,
                               const index_t StrideQ,
                               const index_t StrideK,
                               const index_t StrideV,
                               const index_t StrideO,
                               const index_t BatchStrideQ,
                               const index_t BatchStrideK,
                               const index_t BatchStrideV,
                               const index_t BatchStrideO) const
    {
        // divide problem
        const index_t num_tile_m0 = M0 / kM0PerBlock;    // 4096/128=32 ， 即 given problem[M0, N0] ， 在 M0 方向 需要 32个 t-block
        const index_t num_tile_n1 = N1 / kN1PerBlock;   // 128/128=1 ， 在 N0 方向需要 1个 t-block 

        const index_t id_block = get_block_id(); // TODO: 有可能 kernel launch 进来的 blocks 大约 32x1 ? 

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;    // 商 
            index_t modulus  = dividend - quotient * divisor;  // 模(余数)

            return make_tuple(quotient, modulus);
        };

        const auto [itmp, id_tile_n]          = f(id_block, num_tile_n1); // [id_block, 0]
        const auto [id_tile_batch, id_tile_m] = f(itmp, num_tile_m0);  // [id_block/32, id_block%32] 

        /*
            核心：将2D 计算任务 映射到 1D block 索引上 (id_block)
            * given problem [M0(4096), N1(128)]
            * per block tile size [kM0PerBlock(128), kN1PerBlock(128)]
            * 划分tiles:
                * num_tile_m0 = M0 // kM0PerBlock  # 32
                * num_tile_n1 = N1 // kN1PerBlock  # 1
                * 即block 总数 32 
            * 2D -> 1D id_block 映射 (line 69, 70)
                1. N1 方向 block_idx == id_tile_n == 0
                2. M0 方向 block_idx == id_tile_m == id_block%32 == id_block  
        */

        const index_t iBatch = __builtin_amdgcn_readfirstlane(id_tile_batch);
        const index_t iM0    = __builtin_amdgcn_readfirstlane(id_tile_m * kM0PerBlock); // id_tile_m 中 1st lane 访问的地址，用来定义 tile-window origin
        const index_t iN1    = __builtin_amdgcn_readfirstlane(id_tile_n * kN1PerBlock);  // 0 
        /*
            iM0: id_block * 128
            iN1: 0 

        */

        const auto kernel_impl = FlashAttentionFwdImpl<QDataType,
                                                       KDataType,
                                                       VDataType,
                                                       SaccDataType,
                                                       SMPLComputeDataType,
                                                       PDataType,
                                                       OaccDataType,
                                                       ODataType,
                                                       kBlockSize,
                                                       kHeadDim,
                                                       kM0PerBlock,
                                                       kN0PerBlock,
                                                       kK0PerBlock,
                                                       kN1PerBlock,
                                                       kK1PerBlock>{};

        kernel_impl(q_ptr + iBatch * BatchStrideQ,
                    k_ptr + iBatch * BatchStrideK,
                    v_ptr + iBatch * BatchStrideV,
                    o_ptr + iBatch * BatchStrideO,
                    M0,
                    N0,
                    K0,
                    N1,
                    StrideQ,
                    StrideK,
                    StrideV,
                    StrideO,
                    iM0,
                    iN1);
    }
};

} // namespace ck_tile
