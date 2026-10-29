#pragma once

namespace deep_gemm::mega_moe {

using W4A8Weights = std::tuple<Tensor, Tensor, float>;

static std::pair<W4A8Weights, W4A8Weights> transform_w4a8_weights(const Pair& w1, const Pair& w2) {
    check_device();
    check_tensor(w1.first, torch::kUInt8);
    check_tensor(w2.first, torch::kUInt8);
    TORCH_CHECK(w1.first.dim() == 3 && w2.first.dim() == 3, "W4A8 weights must be [E,N,K/2]");
    const int e = w1.first.size(0), h = w2.first.size(1), i = w2.first.size(2) * 2;
    TORCH_CHECK(e > 0 && h > 0 && h % 256 == 0 && i > 0 && i % 128 == 0,
                "W4A8 MegaMoE requires H divisible by 256 and I divisible by 128");
    check_shape(w1.first, {e, 2*i, h/2});
    check_shape(w2.first, {e, h, i/2});
    const auto device = w1.first.device();
    for (const auto& w : {w1, w2}) {
        check_tensor(w.second, torch::kUInt8);
        TORCH_CHECK(w.first.device() == device && w.second.device() == device, "weights must share a device");
        check_shape(w.second, {e, w.first.size(1), w.first.size(2) / 16});
    }
    using namespace deep_gemm_preprocess_common;
    const auto first = preprocess_weights_and_scales(PreprocessMode::W4A8_MXFP4_MMA, false,
        interleave_gemm1_rows(w1.first), interleave_gemm1_rows(w1.second));
    const auto second = preprocess_weights_and_scales(PreprocessMode::W4A8_MXFP4_MMA, false,
        w2.first, w2.second);
    return {{first.weight, first.scale, first.inv_scale}, {second.weight, second.scale, second.inv_scale}};
}

static void w4a8_mega_moe(const Tensor& out, const W4A8Weights& w1, const W4A8Weights& w2,
    const py::object& buffer, const std::string& act, std::optional<float> limit,
    std::optional<float> alpha, float situ_beta, float situ_linear_beta,
    std::optional<W4A8Weights> r1, std::optional<W4A8Weights> r2) {
    check_device();
    TORCH_CHECK(field<std::string>(buffer, "precision") == "w4a8", "fp8_fp4_mega_moe requires a w4a8 workspace");
    TORCH_CHECK(!buffer.attr("_prepared_num_tokens").is_none(), "call mega_moe_pre_dispatch before MegaMoE");
    TORCH_CHECK(act == "swiglu" || act == "swiglu_oai" || act == "situ",
                "W4A8 supports swiglu, swiglu_oai and situ");
    TORCH_CHECK(act != "situ" || !limit, "activation_limit is not supported by SiTU");
    TORCH_CHECK(act == "situ" || (situ_beta == 4.0f && situ_linear_beta == 25.0f),
                "SiTU parameters require activation='situ'");
    TORCH_CHECK(act != "situ" || (std::isfinite(situ_beta) && situ_beta > 0 &&
                                 std::isfinite(situ_linear_beta) && situ_linear_beta > 0),
                "situ_beta and situ_linear_beta must be finite and positive");
    TORCH_CHECK(!limit || (std::isfinite(*limit) && *limit >= 0), "activation_limit must be finite and nonnegative");
    TORCH_CHECK(!alpha || (std::isfinite(*alpha) && *alpha > 0), "activation_alpha must be finite and positive");
    TORCH_CHECK(act == "swiglu_oai" || !alpha, "activation_alpha requires swiglu_oai");
    const int tokens = field<int>(buffer, "_prepared_num_tokens"), h = field<int>(buffer, "hidden"),
        i = field<int>(buffer, "intermediate_hidden"), rank = field<int>(buffer, "rank"),
        world = field<int>(buffer, "num_ranks"), experts = field<int>(buffer, "num_experts"),
        groups = experts / world, topk = field<int>(buffer, "num_topk"),
        capacity = field<int>(buffer, "num_max_tokens_per_rank");
    const auto device = field<Tensor>(buffer, "buffer").device();
    check_tensor(out, torch::kBFloat16);
    check_shape(out, {tokens, h});
    TORCH_CHECK(out.device() == device, "output must be on the workspace device");
    TORCH_CHECK(r1.has_value() == r2.has_value(), "both stages must use the same replica partition");
    const auto check = [&](const W4A8Weights& w, int n, int k) {
        const auto& [weight, scales, inv] = w;
        check_tensor(weight, torch::kUInt8); check_tensor(scales, torch::kUInt8);
        TORCH_CHECK(weight.dim() == 3 && weight.size(0) > 0, "W4A8 weights must contain local experts");
        check_shape(weight, {weight.size(0), n, k/2});
        check_shape(scales, {weight.size(0), n/64, k*2});
        TORCH_CHECK(weight.device() == device && scales.device() == device, "weights must be on the workspace device");
        TORCH_CHECK(std::isfinite(inv) && inv > 0, "weight_inv_scale must be finite and positive");
    };
    check(w1, 2*i, h);
    check(w2, h, i);
    const int masters = std::get<0>(w1).size(0);
    TORCH_CHECK(std::get<0>(w2).size(0) == masters, "GEMM1/GEMM2 master counts must match");
    if (r1) {
        check(*r1, 2*i, h);
        check(*r2, h, i);
        TORCH_CHECK(std::get<0>(*r1).size(0) == std::get<0>(*r2).size(0), "GEMM1/GEMM2 replica counts must match");
    }
    TORCH_CHECK(masters + (r1 ? std::get<0>(*r1).size(0) : 0) == groups,
                "master and replica counts must match local physical experts");

    // Token count selects compute geometry without resizing the symmetric pool.
    auto g1 = w4fa8::get_best_configs(ceil_div(tokens * topk, groups), 2*i, h, groups, GemmType::MGroupedNoPad);
    if (const auto bm = field<std::optional<int>>(buffer, "_block_m_override")) {
        g1.block_m = *bm;
        g1.warp_m = std::min(*bm, 64);
    }
    g1.n_expand = 1;
    w4fa8::check_config(g1, 2*i, h, GemmType::MGroupedNoPad);
    auto c = read_dispatch_config(field<py::dict>(buffer, "_cfg"));
    c.block_m = g1.block_m; c.gemm1_n = 2*i;
    c.gemm1_block_n = g1.block_n; c.gemm1_block_k = 128;
    c.gemm1_warp_m = g1.warp_m; c.gemm1_warp_n = 64; c.gemm1_stages = g1.num_stages;
    c.num_threads = w4fa8::get_num_threads(g1); c.num_sms = get_num_sms();
    c.num_pull_sms = integer("MEGA_MOE_PULL_SMS", 6);
    TORCH_CHECK(c.num_pull_sms > 0 && c.num_pull_sms < c.num_sms && c.num_pull_sms % 3 == 0,
                "W4A8 pull CTA count must be a positive multiple of three below the total CTA count");
    c.swiglu_oai = act == "swiglu_oai"; c.swiglu_alpha = alpha.value_or(1.702f);
    c.fused_situ = act == "situ";
    c.situ_beta = situ_beta;
    c.situ_linear_beta = situ_linear_beta;
    const auto& replica1 = r1 ? *r1 : w1;
    const auto& replica2 = r2 ? *r2 : w2;
    const auto scratch = field<Tensor>(buffer, "_bf16_scratch");
    const auto pointers = field<Tensor>(buffer, "buf_ptrs");
    mega_moe_native::w4a8_dispatch(c, pointers, rank, tokens,
        std::get<0>(w1), std::get<1>(w1), std::get<2>(w1),
        std::get<0>(replica1), std::get<1>(replica1), std::get<2>(replica1), scratch, limit.value_or(0));
    const auto a = field<Tensor>(buffer, "_activated"), scales = field<Tensor>(buffer, "_activation_scales");
    mega_moe_native::fp8_mega_moe_quantize_local_pool_kernel(c, pointers, rank, scratch, a, scales);
    auto g2 = w4fa8::get_best_configs(ceil_div(tokens * topk, groups), h, i, groups, GemmType::MGroupedNoPad);
    g2.n_expand = 1;
    const auto meta = field<py::object>(buffer, "_metadata");
    const auto combine = field<Tensor>(buffer, "_combine");
    mega_moe_native::w4a8_scatter(g2, groups, a, scales,
        std::get<0>(w2), std::get<1>(w2), std::get<2>(w2),
        std::get<0>(replica2), std::get<1>(replica2), std::get<2>(replica2),
        field<Tensor>(meta, "m_rows"), field<Tensor>(meta, "valid_m_rows"), field<Tensor>(meta, "token_meta"),
        combine, field<Tensor>(buffer, "_combine_offsets"), capacity, field<OptTensor>(buffer, "_gemm2_block_info"));
    mega_moe_aux::recycle_dispatch_counts(field<Tensor>(buffer, "_dispatch_counts"));
    mega_moe_aux::barrier_reduce_impl(combine, field<Tensor>(buffer, "_routing_weights").narrow(0, 0, tokens),
        tokens, h, topk, capacity, field<Tensor>(buffer, "_barrier_pointers"), rank, world,
        field<Tensor>(buffer, "_barrier_done"), std::nullopt, 0, out);
}

} // namespace deep_gemm::mega_moe
