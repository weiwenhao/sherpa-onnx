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
constexpr float kSwooshUnderflowThreshold = -30.0f;
constexpr float kSwooshSlope = 0.08f;

inline float SwooshScalar(float x, float offset, float bias) {
  float t = x - offset;
  float g = t > kSwooshLinearThreshold ? t : std::log1p(std::exp(t));
  return g - kSwooshSlope * x - bias;
}

#if SHERPA_ONNX_SWOOSH_NEON

// exp(r)，r 已归约到 [-ln2/2, ln2/2]，Taylor 到 r^6，相对误差约 1e-7。
inline float32x4_t SwooshExpNeon(float32x4_t x) {
  const float32x4_t inv_ln2 = vdupq_n_f32(1.44269504088896341f);
  const float32x4_t ln2_hi = vdupq_n_f32(0.693359375f);
  const float32x4_t ln2_lo = vdupq_n_f32(-2.12194440e-4f);
  const float32x4_t one = vdupq_n_f32(1.0f);

  float32x4_t n = vrndnq_f32(vmulq_f32(x, inv_ln2));
  float32x4_t r = vfmsq_f32(x, n, ln2_hi);
  r = vfmsq_f32(r, n, ln2_lo);

  float32x4_t p = vfmaq_f32(vdupq_n_f32(1.0f / 120), vdupq_n_f32(1.0f / 720), r);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 24), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 6), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
  p = vfmaq_f32(one, p, r);
  p = vfmaq_f32(one, p, r);

  int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

// log(v)，要求 v >= 1（这里 v = 1 + exp(t) 恒成立）。
// 分解 v = m * 2^e，再用 2*atanh((m-1)/(m+1)) 展开到 s^7。
inline float32x4_t SwooshLogNeon(float32x4_t v) {
  const int32x4_t bias127 = vdupq_n_s32(127);
  const float32x4_t one = vdupq_n_f32(1.0f);

  int32x4_t bits = vreinterpretq_s32_f32(v);
  int32x4_t e = vsubq_s32(
      vshrq_n_s32(vandq_s32(bits, vdupq_n_s32(0x7F800000)), 23), bias127);
  int32x4_t mb = vorrq_s32(vandq_s32(bits, vdupq_n_s32(0x007FFFFF)),
                           vshlq_n_s32(bias127, 23));
  float32x4_t m = vreinterpretq_f32_s32(mb);  // [1, 2)

  // m > sqrt(2) 时折半，把 m 收到 [0.707, 1.414)，s 收到 |s| < 0.172
  uint32x4_t big = vcgtq_f32(m, vdupq_n_f32(1.41421356237f));
  m = vbslq_f32(big, vmulq_f32(m, vdupq_n_f32(0.5f)), m);
  e = vaddq_s32(e, vreinterpretq_s32_u32(vshrq_n_u32(big, 31)));

  float32x4_t s = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
  float32x4_t s2 = vmulq_f32(s, s);
  float32x4_t p = vfmaq_f32(vdupq_n_f32(1.0f / 5), vdupq_n_f32(1.0f / 7), s2);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 3), p, s2);
  p = vfmaq_f32(one, p, s2);
  float32x4_t log_m = vmulq_f32(vdupq_n_f32(2.0f), vmulq_f32(s, p));

  return vfmaq_f32(log_m, vcvtq_f32_s32(e), vdupq_n_f32(0.693147180559945f));
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
  const float32x4_t v_thr = vdupq_n_f32(kSwooshLinearThreshold);
  const float32x4_t v_under = vdupq_n_f32(kSwooshUnderflowThreshold);
  const float32x4_t one = vdupq_n_f32(1.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t x = vld1q_f32(in + i);
    float32x4_t t = vsubq_f32(x, v_offset);
    // 双向夹住再算 exp：上界保证不溢出（t > 阈值的道随后被 bsl 丢弃），
    // 下界保证指数位不下溢。两端夹住后的结果与原图逐位一致。
    float32x4_t tc = vmaxq_f32(vminq_f32(t, v_thr), v_under);
    float32x4_t g = SwooshLogNeon(vaddq_f32(SwooshExpNeon(tc), one));
    g = vbslq_f32(vcgtq_f32(t, v_thr), t, g);
    vst1q_f32(out + i, vsubq_f32(vfmsq_f32(g, x, v_slope), v_bias));
  }
#endif
  for (; i < n; ++i) out[i] = SwooshScalar(in[i], offset, bias);
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SWOOSH_KERNEL_H_
