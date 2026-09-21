// Fused Swoosh activation kernel (NEON).
//
// ZipVoice 的 Zipformer 用 SwooshR/SwooshL 激活，ONNX 导出后是 9~11 个逐元素
// 算子串联（Sub/Mul/Exp/Add/Log/Equal/Where/Sub/Sub），每个算子都要完整遍历
// 一遍激活张量。实测这条链占 decoder 25% 的时间、31% 的能耗。
// 这里把整条链压成一次遍历。
//
//   swoosh(x) = log(1 + exp(x - offset)) - 0.08 * x - bias
//     SwooshR: offset = 1.0, bias = 0.313261687
//     SwooshL: offset = 4.0, bias = 0.035
//
// 必须向量化：ORT 的 Exp/Log 走 MLAS 向量化路径，用标量 expf/logf 写的融合
// 算子会比不融合更慢（实测过 Softplus 重写，慢 66%）。
#ifndef SHERPA_ONNX_CSRC_SWOOSH_KERNEL_H_
#define SHERPA_ONNX_CSRC_SWOOSH_KERNEL_H_

#include <cmath>
#include <cstddef>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SHERPA_ONNX_SWOOSH_NEON 1
#include <arm_neon.h>
#endif

namespace sherpa_onnx {

// x - offset 超过这个阈值后 log(1+exp(t)) 在 fp32 下等于 t（exp(t)+1 == exp(t)），
// 原图用 Where(g == inf, t, g) 表达同一件事。取 30 而非 88 是为了让 exp 永不溢出。
constexpr float kSwooshLinearThreshold = 30.0f;
// t 低于这个值时 exp(t) < 1e-13，fp32 下 1 + exp(t) == 1，log 结果恒为 0；
// 原图也是这个结果。夹住下界是为了避免 exp 的指数位下溢（n < -127 时
// 2^n 的位拼装会产生垃圾）。
constexpr float kSwooshUnderflowThreshold = -16.0f;
constexpr float kSwooshSlope = 0.08f;

inline float SwooshScalar(float x, float offset, float bias) {
  float t = x - offset;
  float g = t > kSwooshLinearThreshold ? t : std::log1p(std::exp(t));
  return g - kSwooshSlope * x - bias;
}

#if SHERPA_ONNX_SWOOSH_NEON

// exp(x)，x ∈ [-16, 0]
static inline float32x4_t SwooshExpNeon(float32x4_t x) {
  const float32x4_t inv_ln2 = vdupq_n_f32(1.44269504f);
  const float32x4_t ln2_hi = vdupq_n_f32(0.693359375f);
  const float32x4_t ln2_lo = vdupq_n_f32(-2.12194440e-4f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  float32x4_t n = vrndnq_f32(vmulq_f32(x, inv_ln2));
  float32x4_t r = vfmsq_f32(x, n, ln2_hi);
  r = vfmsq_f32(r, n, ln2_lo);
  // exp(r), |r| <= ln2/2，Taylor 到 r^4 足够（误差 ~1e-6，远小于 int8 粒度 4e-3）
  float32x4_t p = vfmaq_f32(vdupq_n_f32(1.0f/24), vdupq_n_f32(1.0f/120), r);
  p = vfmaq_f32(vdupq_n_f32(1.0f/6), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
  p = vfmaq_f32(one, p, r);
  p = vfmaq_f32(one, p, r);
  int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

// log(1+v) = v * P(v)，v ∈ [0, 1]。P 是 Chebyshev 节点上拟合的 5 次多项式，
// 最大绝对误差 6.0e-6——int8 量化粒度约 4e-3，留了 400 倍余量。
static inline float32x4_t SwooshLog1pNeon(float32x4_t v) {
  float32x4_t p = vfmaq_f32(vdupq_n_f32(0.10150119f), vdupq_n_f32(-0.02397985f), v);
  p = vfmaq_f32(vdupq_n_f32(-0.21029522f), p, v);
  p = vfmaq_f32(vdupq_n_f32(0.32529598f), p, v);
  p = vfmaq_f32(vdupq_n_f32(-0.49937278f), p, v);
  p = vfmaq_f32(vdupq_n_f32(0.99999184f), p, v);
  return vmulq_f32(v, p);
}

#endif  // SHERPA_ONNX_SWOOSH_NEON

// out[i] = log(1 + exp(in[i] - offset)) - 0.08 * in[i] - bias
inline void SwooshForward(const float *in, float *out, size_t n, float offset,
                          float bias) {
  size_t i = 0;
#if SHERPA_ONNX_SWOOSH_NEON
  const float32x4_t v_offset = vdupq_n_f32(offset);
  const float32x4_t v_bias = vdupq_n_f32(bias);
  const float32x4_t v_slope = vdupq_n_f32(kSwooshSlope);
  const float32x4_t v_lo = vdupq_n_f32(kSwooshUnderflowThreshold);
  const float32x4_t v_zero = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t x = vld1q_f32(in + i);
    float32x4_t t = vsubq_f32(x, v_offset);
    // softplus(t) = max(t,0) + log(1 + exp(-|t|))
    // 这样 exp 的参数恒为负（夹到 -16 下界防指数位下溢），log 只需在 [0,1] 上
    // 用多项式近似，省掉通用 log() 的指数位拆解、fdiv 和高次级数。
    float32x4_t v = SwooshExpNeon(vmaxq_f32(vnegq_f32(vabsq_f32(t)), v_lo));
    float32x4_t g = vaddq_f32(vmaxq_f32(t, v_zero), SwooshLog1pNeon(v));
    vst1q_f32(out + i, vsubq_f32(vfmsq_f32(g, x, v_slope), v_bias));
  }
#endif
  for (; i < n; ++i) out[i] = SwooshScalar(in[i], offset, bias);
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SWOOSH_KERNEL_H_
