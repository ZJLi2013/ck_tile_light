// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

#include "tile_gemm_shape.hpp"
#include "../../../example/ck_tile/99_toy_example/02_gemm/block_gemm_pipeline_agmem_bgmem_creg.hpp"

#include "block_gemm_pipeline_agmem_bgmem_creg_v2_askiplds.hpp"
#include "block_gemm_pipeline_problem.hpp"
#include "block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck_tile/ops/reduce.hpp"


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
struct FlashAttentionFwdImpl
{
    // block gemm0 pipeline
    using BlockGemm0Problem = BlockGemmPipelineProblem<
        QDataType,
        KDataType,
        SaccDataType,
        kBlockSize,
        TileGemmShape<kM0PerBlock, kN0PerBlock, kK0PerBlock>>;

    using BlockGemm0Policy =
        BlockGemmPipelineAGmemBGmemCRegSkipALdsPersistentQRegCachePolicy<kHeadDim>;

    using BlockGemm0Pipeline =
        BlockGemmPipelineAGmemBGmemCReg<BlockGemm0Problem, BlockGemm0Policy>;

    // block gemm1
    using BlockGemm1 = BlockGemmARegBSmemCRegV1<
        BlockGemmARegBSmemCRegProblem<
            PDataType,
            VDataType,
            OaccDataType,
            kBlockSize,
            TileGemmShape<kM0PerBlock, kN1PerBlock, kK1PerBlock>>,
        BlockGemmARegBSmemCRegV1DefaultPolicy>;

    // 3d, with padding
    __device__ static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = kN1PerBlock;  // 128
        constexpr index_t kKPerBlock = kK1PerBlock;  // 32 
        constexpr index_t kPad       = 1;
        // 2% faster than use kK1 = 8
        constexpr index_t kK1 = 8;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kK1>{}, number<kNPerBlock>{}, number<kK1>{}),
            make_tuple(number<(kNPerBlock + kPad) * kK1>{}, number<kK1>{}, number<1>{}),
            number<kK1>{},
            number<1>{});
        /*
            [128, 32] ->  [4, 128, 8]  { kKPerBlock/8,  kNPerBlock, 8 }
        */

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(number<kKPerBlock / kK1>{}, number<kK1>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        
        /*
            { kKPerBlock/8,  kNPerBlock, 8 } -> 写出layout: { kNPerBlock, kKPerBlock }
            即 v_lds [128, 32]
        */

        return b_lds_block_desc;
    }

    __device__ static constexpr auto MakeVDramTileDistribution()
    {
        using BDataType = VDataType;

        constexpr index_t kNPerBlock = kN1PerBlock;
        constexpr index_t kKPerBlock = kK1PerBlock;

        constexpr index_t K1 = 16 / sizeof(BDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
    }

    __device__ static constexpr index_t GetStaticLdsSize()
    {
        return max(BlockGemm0Pipeline::GetStaticLdsSize(),
                   static_cast<index_t>(MakeVLdsBlockDescriptor().get_element_space_size() *
                                        sizeof(VDataType)));
    }

    __device__ void operator()(const QDataType* q_ptr,
                               const KDataType* k_ptr,
                               const VDataType* v_ptr,
                               ODataType* o_ptr,
                               const index_t M0,
                               const index_t N0,
                               const index_t K0,
                               const index_t N1,
                               const index_t StrideQ,
                               const index_t StrideK,
                               const index_t StrideV,
                               const index_t StrideO,
                               const index_t iM0,
                               const index_t iN1) const
    {
        constexpr auto I0 = number<0>{};
        constexpr auto I1 = number<1>{};

        // allocate LDS
        __shared__ char smem_ptr[GetStaticLdsSize()];

        // Q/K/V DRAM and DRAM window
        const auto q_dram = make_naive_tensor_view<address_space_enum::global>(
            q_ptr, make_tuple(M0, kHeadDim), make_tuple(StrideQ, 1), number<32>{}, number<1>{});
            /*  M0: q_seqlen
                kHeadDim: K0
             */

        const auto k_dram = make_naive_tensor_view<address_space_enum::global>(
            k_ptr, make_tuple(N0, K0), make_tuple(StrideK, 1), number<32>{}, number<1>{});
            /*
                N0: kv_seqlen
                K0: q_headdim
                k_ptr[kv_seqlen, kv_headdim]  

            */

        const auto v_dram = make_naive_tensor_view<address_space_enum::global>(
            v_ptr, make_tuple(N1, N0), make_tuple(StrideV, 1), number<32>{}, number<1>{});
            /*
                N1: kv_headdim 
                v_ptr[kv_headim, kv_seqlen] 
                StrideV:: N0 
                TODO: 不应该是 [kv_seqlen, kv_headim] ??
            */

        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<kM0PerBlock>{}, number<kHeadDim>{}),
            {iM0, 0},
            BlockGemm0Policy::template MakeADramTileDistribution<BlockGemm0Problem>());
        /*
            * q_dram [seqlen_q(4096), q_headsize(128)]
            * q_dram_window(tblock-level), [kM0PerBlock(128), kHeadDim(32)] ， tblock 之间偏移 {iM0, 0}
            * lane-data mapping 由MakeADramTileDistribution() 定义，k 方向 loops 也定义在 ADramTileDistr 中 ..  
                * 也即，ADramTileDist 实际上定义了对  [128, 128] Q_tile 的访问，但是 q_dram_window 本身的shape 是 [128, 32]
        */

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<kN0PerBlock>{}, number<kK0PerBlock>{}), {0, 0});
        /*
            k_dram[seqlen_kv(4096), kv_headsize(128)]: kv_hd 方向由 same block loopover，
            k_dram_window [kN0PerBlock(128), kK0PerBlock(32)]。 
            
            1. 实际计算用的 k 转置，即 [32, 128] 
            2. 这里并没有使用 MakeBDramTileDistribution ? 
            3. tblock 之间的偏移 {0, 0} 
                * make_tile_window() 中偏移只是 t-block level的，而 tensor K 沿行方向 是同一个 tblock loopover，所以偏移就是{0, 0}
            
            * 整体上就有两层loop，见 动图
                1. inter-block [128, 128] 滑动，将 given problem 分解到 tblock 上
                2. intra-block [128, 32] 沿 k loop 4 次 分解 [128, 128] 
        */

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(number<kN1PerBlock>{}, number<kK1PerBlock>{}),
                             {iN1, 0}, // {0, 0}
                             MakeVDramTileDistribution());

        /*
            v_dram[N1(kv_head), N0(seqlen_kv)] # [128, 4096] 
            v_dram_window [kN1PerBlock(128), kK1PerBlock(32)]， 
            * tblock 之间的偏移 {0, 0}
                * 同理k, tensor v 的acccess 是 intra-block loopover
            * v_dram 的layout 有点诡异啊，常规理解应该是 v_dram[seqlen_kv, kv_hz]，但是合理：
                * P_tile[128, 128] =  q_dram_window[128, 32] * k_dram_window[128, 32].T 
                * O_tile =  P_tile[128, 128] * v_dram_window[128, 32] 
        */

        // Q in register
        auto q_reg_tensor = load_tile(q_dram_window);
        /* cold loop 后 cache q[128, 128] */

        // V LDS and LDS window
        // V LDS occupies the same LDS allocation Q/K LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(reinterpret_cast<VDataType*>(smem_ptr),
                                                               MakeVLdsBlockDescriptor());
        /*
            * v_lds [kN1PerBlock(128), kK1PerBlock(32)] 
        */

        auto v_lds_window = make_tile_window(
            v_lds, make_tuple(number<kN1PerBlock>{}, number<kK1PerBlock>{}), {0, 0});

        /*
            * v_lds_window [kN1PerBlock(128), kK1PerBlock(32)]， 与 v_dram_window 一致
            * v_lds 是 gemm1 中 b 矩阵
        */

        // Block GEMM0 pipeline and Block GEMM1
        constexpr auto gemm0_pipeline = BlockGemm0Pipeline{};
        constexpr auto gemm1          = BlockGemm1{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SaccBlockTileType =
            decltype(gemm0_pipeline(q_dram_window, k_dram_window, q_reg_tensor, nullptr));

        using SBlockTileType = decltype(tile_elementwise_in(
            type_convert<SMPLComputeDataType, SaccDataType>, SaccBlockTileType{}));

        using PBlockTileType = decltype(tile_elementwise_in(type_convert<PDataType, SaccDataType>,
                                                            SaccBlockTileType{}));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm1(
            get_slice_tile(
                PBlockTileType{}, sequence<0, 0>{}, sequence<kM0PerBlock, kK1PerBlock>{}),
            v_dram_window));

        // init Sacc, Oacc, M, L
        auto s_acc = SaccBlockTileType{};
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        tile_elementwise_inout([](auto& e) { e = 0; }, o_acc);
        tile_elementwise_inout([](auto& e) { e = std::numeric_limits<SMPLComputeDataType>::lowest(); },
                               m);
        tile_elementwise_inout([](auto& e) { e = 0; }, l);

        // loop over Column of S (J loop) 
        // S[M0, N0] 
        index_t iN0 = 0;

        do
        {
            s_acc = gemm0_pipeline(k_dram_window, q_reg_tensor, smem_ptr);
            /*
                * s_acc = k_dram_window [128, 32] *  q_reg_tensor[128, 128] 

                gemm0_pipeline.operator(k_dram_window, a_reg_tensor, smem_ptr) 中
                    * smem_ptr 用来host k_lds，分别定义  key_copy_lds_window, k_lds_gemm_window
                    * k_copy_lds_window 用于接收 k_dram_window (lds write)
                    * k_lds_gemm_window 用于给mfma 指令做输入（lds read)
                    * gemm0 与 gemm1 使用同样的 blockgemm 实例: BlockGemmARegBSmemCRegV1
                * pipline输出为 c_block_tile，其tile_distr 由 blockgemm 实例返回类型确定

                * gemm0_pipeine 计算流程:                
                    1. 从输入 a_reg_tensor 按 ARegBlockDesc 取值存入 a_copy_reg_tensor 
                    2. 从输入 b_dram_window 写入 b_copy_lds_window
                    3. block_gemm(c_block_tile, get_slice_tile(a_copy_reg_tensor, ..), b_copy_lds_window) 
                
            */

            // S{j}
            const auto s =
                tile_elementwise_in(type_convert<SMPLComputeDataType, SaccDataType>, s_acc); // 转格式

            // prefetch load v tile
            const auto v_prefetch = load_tile(v_dram_window);  // gemm2 先prefetch v, p/s 此时已经存在on-chip memory

            // m_local = rowmax(S{j})
            // m_local:: 当前 block 上 per row 最大值 
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s, sequence<1>{}, f_max, std::numeric_limits<SMPLComputeDataType>::lowest());

            block_tile_reduce_sync(m_local, f_max);

            // m{j-1}
            const auto m_old = m;

            // m{j}
            // m{j}:: 跟新 累计到当前block 为止的所有元素 per row 的最大值    m{j} = max{m_old, m_local}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);

            // Pcompute{j}
            // P: 当前block上元素对分母的贡献， p 与 s 同 tile_distribution 
            auto p_compute =
                make_static_distributed_tensor<SMPLComputeDataType>(s.get_tile_distribution());

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();

            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    p_compute(i_j_idx) = exp(s[i_j_idx] - m[i_idx]);
                });
            });
            /*
                当前block上元素point-wise对分母的贡献， p = exp(s[i] - m[i] )，point-wise op 
            */

            // rowsum(Pcompute{j})
            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0});

            block_tile_reduce_sync(rowsum_p, f_sum);

            /*
                rowsum(pi) 当前block上元素累计对分母的贡献
            */

            // l{j}, Oacc{j}
            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const auto tmp = exp(m_old[i_idx] - m[i_idx]);

                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                /*
                    1. l[i] 累计到当前block 的所有元素对分母的贡献 $ li = \exp(m[i-1] - m[i]) * l[i-1] + rowsum(pi)$
                */

                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    o_acc(i_j_idx) *= tmp;  
                    /*
                        2. 乘以 tmp 是为了将旧分子 o_acc 按比例缩放，匹配新的 m[i]。o_acc(i_j_idx) 本质上是还没加上 p[i]*V[i] 的旧部分，它只需要按 exp(m_old - m) 缩放，不要乘 l[i_idx]。
                    */
                });
            });

            block_sync_lds();
            store_tile(v_lds_window, v_prefetch);
            move_tile_window(v_dram_window, {0, kK1PerBlock}); 
            /*
                1. online softmax 相关参数计算同时， prefetch load v from DRAM :: v_prefetch = load_tile(v_dram_window);
                2. onlien softmax 相关参数计算完成时， write v_lds_window by v_prefetch
                3. 以 v_dram_window[128, 32] 在 v_dram [128, 4096] 上滑动，每次偏移 [0,  32]
            */

            // type cast Pcompute{j} into P{j}
            const auto p =
                tile_elementwise_in(type_convert<PDataType, SMPLComputeDataType>, p_compute); // 数据类型转换

            // Oacc{j}
            constexpr index_t k1_loops = kN0PerBlock / kK1PerBlock;
            /*
                * P_tile[kM0PerBlock(128), kN0PerBlock(128)] =  q_dram_window[kM0(128), kK0(32)] * k_dram_window[kN0(128), kK0(32)d].T 
                * S_tile[128, 128] = softmax(P_tile)
                * O_tile =  S_tile[kM0PerBlock(128), kN0PerBlock(128)]  * v_dram_window[kN1PerBlock(128), kK1PerBlock(32)]

                * 注意， P_tile 在寄存器上，由 blockgemm0 实例确定 (agbg_creg)
                * s_tile 列数是 kN0PerBlock，每次从S_tile 加载的shape 是[128, kK1PerBlock]，故 k1 方向loops 数  kN0PerBlock / kK1PerBlock 
            */

            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    const auto v = load_tile(v_dram_window); // load next v
                    block_sync_lds();
                    gemm1(o_acc,
                          get_slice_tile(p,
                                         sequence<0, i_k1 * kK1PerBlock>{},
                                         sequence<kM0PerBlock, (i_k1 + 1) * kK1PerBlock>{}),
                          v_lds_window);  //  o_acc += p_slice * v_lds 
                    /*
                        O[i] = l[i-1] * exp(m[i-1] - m[i]) * O[i-1] + p[i] * V[j] 计算中的 2nd 部分:  p[i] * V[j]
                    */
                    block_sync_lds();
                    store_tile(v_lds_window, v);
                    move_tile_window(v_dram_window, {0, kK1PerBlock});
                    /*
                         1. load from dram with v_dram_window to v 
                         2. lds write v to v_lds_window 
                         3. 以 v_dram_window[128, 32] 在 v_dram [128, 4096] 上滑动，每次偏移 [0,  32]
                    实际上就是把 v_dram 上一块 [128, 128] 的tile 拿来给当前 tblock 计算了
                    */
                });
            }
            // tail
            {
                block_sync_lds();
                gemm1(o_acc,
                      get_slice_tile(p,
                                     sequence<0, (k1_loops - 1) * kK1PerBlock>{},
                                     sequence<kM0PerBlock, kN0PerBlock>{}),
                      v_lds_window);
                block_sync_lds();
            }
            // move tile windows
            move_tile_window(k_dram_window, {kN0PerBlock, 0});   
            /*
                一个完整的fused gemm0-softmax-gemm1 计算完成后，到下一个 [128, 128] k block with  k_dram[4096, 128] 
                * 注意 k_dram_window[128, 32], gemm0 实际上有 k0_loop=4，实际完成的是 [128, 128] block tile 
            */
            iN0 += kN0PerBlock;
        } while(iN0 < N0);  // loop over col of S by [128, 128]

        // Oacc
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            const auto tmp = 1 / l[i_idx];

            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                o_acc(i_j_idx) *= tmp;
            });
        });

        // type cast Oacc into O
        const auto o = tile_elementwise_in(type_convert<ODataType, OaccDataType>, o_acc);

        // O DRAM and O DRAM window
        auto o_dram = make_naive_tensor_view<address_space_enum::global>(
            o_ptr, make_tuple(M0, N1), make_tuple(StrideO, 1), number<32>{}, number<1>{});

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<kM0PerBlock>{}, number<kN1PerBlock>{}),
                             {iM0, iN1},
                             o.get_tile_distribution());

        // store O
        store_tile(o_dram_window, o);
        /*
            也就是一个 tblock，最终完成了  output gemm 中 [128, 128] tile 的计算。
            为了完成这个 o_tile 计算，实际上需要 loop over K[128, 4096] by [128, 128]，以及 loop over V[4096, 128] by [128, 128]
        */
    }
};

} // namespace ck_tile
