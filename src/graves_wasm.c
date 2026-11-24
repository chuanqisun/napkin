/*
 * Method C: WASM SIMD128 Graves (2013) IAM-OnDB Cursive Handwriting Engine
 * - Pre-cached primed states (0 priming steps!)
 * - Per-row INT8 quantized LSTM weights with 4-row unrolled i32x4.dot_i16x8 SIMD128 kernel
 * - Single JS->WASM call per phrase (zero per-step boundary overhead)
 *
 * Compile:
 *   clang --target=wasm32 -O3 -msimd128 -nostdlib -fno-builtin \
 *     -Wl,--no-entry -Wl,--export=init_offsets -Wl,--export=synthesize \
 *     -Wl,--export=get_weights_ptr -Wl,--export=get_chars_ptr -Wl,--export=get_out_ptr \
 *     -Wl,--initial-memory=10485760 -Wl,--max-memory=16777216 \
 *     -o exports/handwriting/graves_simd.wasm src/graves_wasm.c
 */
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

// Static buffers inside WASM linear memory (zero malloc needed)
// Weight binary is 4.00 MB -> reserve 4.5 MB
#define WEIGHTS_CAP (4600 * 1024)
#define MAX_CHARS 256
#define MAX_STEPS 2048

static uint8_t  WEIGHTS_BUF[WEIGHTS_CAP] __attribute__((aligned(16)));
static int32_t  CHARS_BUF[MAX_CHARS];
static float    OUT_BUF[MAX_STEPS * 3]; // [dx, dy, eos]

// Offsets (byte offsets inside WEIGHTS_BUF) configured by JS from graves_fast_meta.json
static uint32_t OFF_L1_WQ, OFF_L1_WSC, OFF_L1_B;
static uint32_t OFF_L2_WQ, OFF_L2_WSC, OFF_L2_B;
static uint32_t OFF_L3_WQ, OFF_L3_WSC, OFF_L3_B;
static uint32_t OFF_ATTN_WT, OFF_ATTN_B;
static uint32_t OFF_GMM_WT, OFF_GMM_B;
static uint32_t OFF_PRIMED;

// Scratch buffers (16-byte aligned)
static float   H1[400], C1[400], H2[400], C2[400], H3[400], C3[400];
static float   KAPPA[10], W_CTX[73];
static float   IN_F32[896] __attribute__((aligned(16)));
static int16_t IN_I16[896] __attribute__((aligned(16)));
static float   GATES[1600] __attribute__((aligned(16)));
static float   ATTN_OUT[32] __attribute__((aligned(16)));
static float   PHI[MAX_CHARS];
static float   GMM_OUT[128] __attribute__((aligned(16)));

uint32_t get_weights_ptr(void) { return (uint32_t)(uintptr_t)WEIGHTS_BUF; }
uint32_t get_chars_ptr(void)   { return (uint32_t)(uintptr_t)CHARS_BUF; }
uint32_t get_out_ptr(void)     { return (uint32_t)(uintptr_t)OUT_BUF; }

void init_offsets(
    uint32_t l1_wq, uint32_t l1_wsc, uint32_t l1_b,
    uint32_t l2_wq, uint32_t l2_wsc, uint32_t l2_b,
    uint32_t l3_wq, uint32_t l3_wsc, uint32_t l3_b,
    uint32_t attn_wt, uint32_t attn_b,
    uint32_t gmm_wt, uint32_t gmm_b,
    uint32_t primed
) {
    OFF_L1_WQ = l1_wq; OFF_L1_WSC = l1_wsc; OFF_L1_B = l1_b;
    OFF_L2_WQ = l2_wq; OFF_L2_WSC = l2_wsc; OFF_L2_B = l2_b;
    OFF_L3_WQ = l3_wq; OFF_L3_WSC = l3_wsc; OFF_L3_B = l3_b;
    OFF_ATTN_WT = attn_wt; OFF_ATTN_B = attn_b;
    OFF_GMM_WT = gmm_wt; OFF_GMM_B = gmm_b;
    OFF_PRIMED = primed;
}

// Fast accurate math primitives (no libm dependency)
static inline float fast_expf(float x) {
    if (x < -80.0f) return 0.0f;
    if (x > 80.0f) x = 80.0f;
    // Range reduction: e^x = 2^n * e^r, r in [-ln2/2, ln2/2]
    const float LOG2E = 1.44269504089f;
    const float LN2_HI = 0.693359375f;
    const float LN2_LO = -2.12194440e-4f;
    float fx = x * LOG2E;
    int32_t n = (int32_t)(fx + (fx >= 0.0f ? 0.5f : -0.5f));
    float fn = (float)n;
    float r = (x - fn * LN2_HI) - fn * LN2_LO;
    // Degree-5 minimax polynomial for e^r
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.04166667f + r * 0.00833333f))));
    // Multiply by 2^n via IEEE-754 exponent
    union { float f; uint32_t u; } pow2;
    pow2.u = (uint32_t)(n + 127) << 23;
    return p * pow2.f;
}

static inline float fast_logf(float x) {
    if (x <= 1e-30f) return -69.07755f;
    union { float f; uint32_t u; } v;
    v.f = x;
    int32_t e = (int32_t)((v.u >> 23) & 0xFF) - 127;
    v.u = (v.u & 0x007FFFFFu) | 0x3F800000u; // m in [1, 2)
    float m = v.f;
    // Padé / series around m=1 using y = (m-1)/(m+1)
    float y = (m - 1.0f) / (m + 1.0f);
    float y2 = y * y;
    float ln_m = 2.0f * y * (1.0f + y2 * (0.33333333f + y2 * (0.2f + y2 * 0.14285714f)));
    return ln_m + (float)e * 0.69314718056f;
}

static inline float fast_sigmoid(float x) {
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) return 1.0f;
    return 1.0f / (1.0f + fast_expf(-x));
}

static inline float fast_tanh(float x) {
    if (x < -10.0f) return -1.0f;
    if (x > 10.0f) return 1.0f;
    float e2 = fast_expf(2.0f * x);
    return (e2 - 1.0f) / (e2 + 1.0f);
}

static inline float fast_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return fast_logf(1.0f + fast_expf(x));
}

// Fast xorshift32 + Box-Muller PRNG
static uint32_t rng_state = 123456789u;
static inline float rng_uniform(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return (float)(x & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

static inline float fast_cosf(float x) {
    // Reduce to [-pi, pi]
    const float TWO_PI = 6.28318530718f;
    const float PI = 3.14159265359f;
    int32_t k = (int32_t)(x / TWO_PI);
    x -= (float)k * TWO_PI;
    if (x > PI) x -= TWO_PI;
    if (x < -PI) x += TWO_PI;
    // Reflect to [-pi/2, pi/2]
    float sign = 1.0f;
    if (x > 1.5707963f) { x = PI - x; sign = -1.0f; }
    else if (x < -1.5707963f) { x = -PI - x; sign = -1.0f; }
    float x2 = x * x;
    return sign * (1.0f + x2 * (-0.5f + x2 * (0.04166667f - x2 * 0.00138889f)));
}

static inline float fast_sinf(float x) {
    return fast_cosf(x - 1.57079632679f);
}

static inline void rng_normal2(float *z1, float *z2) {
    float u1 = rng_uniform();
    if (u1 < 1e-6f) u1 = 1e-6f;
    float u2 = rng_uniform();
    float r = __builtin_sqrtf(-2.0f * fast_logf(u1));
    float theta = 6.28318530718f * u2;
    *z1 = r * fast_cosf(theta);
    *z2 = r * fast_sinf(theta);
}

// Horizontal sum of 4 int32 lanes
static inline int32_t hsum_i32x4(v128_t v) {
    return wasm_i32x4_extract_lane(v, 0) + wasm_i32x4_extract_lane(v, 1)
         + wasm_i32x4_extract_lane(v, 2) + wasm_i32x4_extract_lane(v, 3);
}

// Horizontal sum of 4 float32 lanes
static inline float hsum_f32x4(v128_t v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1)
         + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

/*
 * 4-row unrolled INT8xINT16 SIMD128 matvec using wasm_i32x4_dot_i16x8:
 *   W_q: [out_dim, in_pad] int8 (in_pad % 16 == 0, out_dim % 4 == 0)
 *   IN_F32[0..in_dim-1] quantized once to IN_I16[0..in_pad-1] in [-127..127]
 */
static void matvec_int8_simd(
    const float *x_f32, int in_dim, int in_pad, int out_dim,
    uint32_t off_wq, uint32_t off_wsc, uint32_t off_b,
    float *out
) {
    // 1. Dynamic symmetric INT8 quantization of x into int16_t buffer (fits in L1 cache!)
    float max_abs = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        float a = x_f32[i] >= 0.0f ? x_f32[i] : -x_f32[i];
        if (a > max_abs) max_abs = a;
    }
    float x_scale = max_abs > 1e-8f ? (max_abs / 127.0f) : 1.0f;
    float inv_sc = 1.0f / x_scale;
    for (int i = 0; i < in_dim; i++) {
        float v = x_f32[i] * inv_sc;
        int32_t q = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        else if (q < -127) q = -127;
        IN_I16[i] = (int16_t)q;
    }
    for (int i = in_dim; i < in_pad; i++) IN_I16[i] = 0;

    const int8_t *wq_base = (const int8_t *)(WEIGHTS_BUF + off_wq);
    const float  *wsc     = (const float  *)(WEIGHTS_BUF + off_wsc);
    const float  *bias    = (const float  *)(WEIGHTS_BUF + off_b);

    // 2. 4-row unrolled SIMD dot product
    for (int r = 0; r < out_dim; r += 4) {
        const int8_t *w0 = wq_base + (size_t)(r + 0) * in_pad;
        const int8_t *w1 = wq_base + (size_t)(r + 1) * in_pad;
        const int8_t *w2 = wq_base + (size_t)(r + 2) * in_pad;
        const int8_t *w3 = wq_base + (size_t)(r + 3) * in_pad;

        v128_t acc0 = wasm_i32x4_splat(0);
        v128_t acc1 = wasm_i32x4_splat(0);
        v128_t acc2 = wasm_i32x4_splat(0);
        v128_t acc3 = wasm_i32x4_splat(0);

        for (int c = 0; c < in_pad; c += 16) {
            v128_t x_lo = wasm_v128_load(&IN_I16[c]);
            v128_t x_hi = wasm_v128_load(&IN_I16[c + 8]);

            v128_t v0 = wasm_v128_load(&w0[c]);
            acc0 = wasm_i32x4_add(acc0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(v0), x_lo));
            acc0 = wasm_i32x4_add(acc0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(v0), x_hi));

            v128_t v1 = wasm_v128_load(&w1[c]);
            acc1 = wasm_i32x4_add(acc1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(v1), x_lo));
            acc1 = wasm_i32x4_add(acc1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(v1), x_hi));

            v128_t v2 = wasm_v128_load(&w2[c]);
            acc2 = wasm_i32x4_add(acc2, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(v2), x_lo));
            acc2 = wasm_i32x4_add(acc2, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(v2), x_hi));

            v128_t v3 = wasm_v128_load(&w3[c]);
            acc3 = wasm_i32x4_add(acc3, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(v3), x_lo));
            acc3 = wasm_i32x4_add(acc3, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(v3), x_hi));
        }

        out[r + 0] = (float)hsum_i32x4(acc0) * (wsc[r + 0] * x_scale) + bias[r + 0];
        out[r + 1] = (float)hsum_i32x4(acc1) * (wsc[r + 1] * x_scale) + bias[r + 1];
        out[r + 2] = (float)hsum_i32x4(acc2) * (wsc[r + 2] * x_scale) + bias[r + 2];
        out[r + 3] = (float)hsum_i32x4(acc3) * (wsc[r + 3] * x_scale) + bias[r + 3];
    }
}

// FP32 SIMD128 matvec for small layers (attn 30x480, gmm 121x400)
static void matvec_f32_simd(
    const float *x, int in_pad, int out_dim,
    uint32_t off_wt, uint32_t off_b,
    float *out
) {
    const float *wt   = (const float *)(WEIGHTS_BUF + off_wt);
    const float *bias = (const float *)(WEIGHTS_BUF + off_b);
    for (int r = 0; r < out_dim; r++) {
        const float *w = wt + (size_t)r * in_pad;
        v128_t acc = wasm_f32x4_splat(0.0f);
        for (int c = 0; c < in_pad; c += 4) {
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(&w[c]), wasm_v128_load(&x[c])));
        }
        out[r] = hsum_f32x4(acc) + bias[r];
    }
}

static void lstm_update(float *h, float *c, const float *gates) {
    for (int i = 0; i < 400; i++) {
        float gi = fast_sigmoid(gates[i]);
        float gj = fast_tanh(gates[400 + i]);
        float gf = fast_sigmoid(gates[800 + i] + 1.0f);
        float go = fast_sigmoid(gates[1200 + i]);
        float cn = c[i] * gf + gi * gj;
        c[i] = cn;
        h[i] = fast_tanh(cn) * go;
    }
}

/*
 * Run full cached-state Graves synthesis inside WASM.
 * Inputs:
 *   CHARS_BUF[0..U-1]: alphabet indices of (tail_text + text + " " + "\0")
 * Returns: number of generated points written to OUT_BUF[0..3*n-1]
 */
int32_t synthesize(int32_t style_id, int32_t U, float bias, int32_t max_steps, uint32_t seed) {
    if (style_id < 0 || style_id > 12) style_id = 9;
    if (U > MAX_CHARS) U = MAX_CHARS;
    if (max_steps > MAX_STEPS) max_steps = MAX_STEPS;
    rng_state = seed ? seed : 42u;

    // 1. Load pre-cached primed state (2483 floats)
    const float *ps = (const float *)(WEIGHTS_BUF + OFF_PRIMED) + (size_t)style_id * 2483;
    for (int i = 0; i < 400; i++) {
        H1[i] = ps[i];
        C1[i] = ps[400 + i];
        H2[i] = ps[800 + i];
        C2[i] = ps[1200 + i];
        H3[i] = ps[1600 + i];
        C3[i] = ps[2000 + i];
    }
    for (int i = 0; i < 10; i++) KAPPA[i] = ps[2400 + i];
    for (int i = 0; i < 73; i++) W_CTX[i] = ps[2410 + i];

    float x0 = 0.0f, x1 = 0.0f, x2 = 1.0f;
    int32_t out_n = 0;

    for (int step = 0; step < max_steps; step++) {
        // --- LSTM 1: [w(73), x(3), h1(400)] = 476 ---
        for (int i = 0; i < 73; i++)  IN_F32[i] = W_CTX[i];
        IN_F32[73] = x0; IN_F32[74] = x1; IN_F32[75] = x2;
        for (int i = 0; i < 400; i++) IN_F32[76 + i] = H1[i];
        matvec_int8_simd(IN_F32, 476, 480, 1600, OFF_L1_WQ, OFF_L1_WSC, OFF_L1_B, GATES);
        lstm_update(H1, C1, GATES);

        // --- Attention: [w(73), x(3), new_h1(400)] = 476 (pad 480) ---
        for (int i = 0; i < 400; i++) IN_F32[76 + i] = H1[i];
        IN_F32[476] = 0.0f; IN_F32[477] = 0.0f; IN_F32[478] = 0.0f; IN_F32[479] = 0.0f;
        matvec_f32_simd(IN_F32, 480, 30, OFF_ATTN_WT, OFF_ATTN_B, ATTN_OUT);

        float alpha[10], inv_beta[10];
        for (int k = 0; k < 10; k++) {
            alpha[k] = fast_softplus(ATTN_OUT[k]);
            float b = fast_softplus(ATTN_OUT[10 + k]);
            if (b < 0.01f) b = 0.01f;
            inv_beta[k] = 1.0f / b;
            KAPPA[k] += fast_softplus(ATTN_OUT[20 + k]) * (1.0f / 25.0f);
        }

        // Compute phi[u] and sparse one-hot lookup into W_CTX[73]
        for (int c = 0; c < 73; c++) W_CTX[c] = 0.0f;
        int best_u = 0;
        float best_phi = -1.0f;
        for (int u = 0; u < U; u++) {
            float fu = (float)u;
            float s = 0.0f;
            for (int k = 0; k < 10; k++) {
                float d = KAPPA[k] - fu;
                float arg = -(d * d) * inv_beta[k];
                if (arg > -25.0f) s += alpha[k] * fast_expf(arg);
            }
            PHI[u] = s;
            if (s > best_phi) { best_phi = s; best_u = u; }
            int32_t ch = CHARS_BUF[u];
            if (ch >= 0 && ch < 73) W_CTX[ch] += s;
        }

        // --- LSTM 2: [x(3), h1(400), w(73), h2(400)] = 876 ---
        IN_F32[0] = x0; IN_F32[1] = x1; IN_F32[2] = x2;
        for (int i = 0; i < 400; i++) IN_F32[3 + i] = H1[i];
        for (int i = 0; i < 73; i++)  IN_F32[403 + i] = W_CTX[i];
        for (int i = 0; i < 400; i++) IN_F32[476 + i] = H2[i];
        matvec_int8_simd(IN_F32, 876, 880, 1600, OFF_L2_WQ, OFF_L2_WSC, OFF_L2_B, GATES);
        lstm_update(H2, C2, GATES);

        // --- LSTM 3: [x(3), h2(400), w(73), h3(400)] = 876 ---
        for (int i = 0; i < 400; i++) IN_F32[3 + i] = H2[i];
        for (int i = 0; i < 400; i++) IN_F32[476 + i] = H3[i];
        matvec_int8_simd(IN_F32, 876, 880, 1600, OFF_L3_WQ, OFF_L3_WSC, OFF_L3_B, GATES);
        lstm_update(H3, C3, GATES);

        // --- GMM head: h3(400) -> 121 ---
        matvec_f32_simd(H3, 400, 121, OFF_GMM_WT, OFF_GMM_B, GMM_OUT);

        // Sample 20-component MDN
        float pis[20];
        float max_pi = -1e30f;
        float b1 = 1.0f + bias;
        for (int k = 0; k < 20; k++) {
            pis[k] = GMM_OUT[k] * b1;
            if (pis[k] > max_pi) max_pi = pis[k];
        }
        float sum_pi = 0.0f;
        for (int k = 0; k < 20; k++) {
            pis[k] = fast_expf(pis[k] - max_pi);
            sum_pi += pis[k];
        }
        float sum2 = 0.0f;
        for (int k = 0; k < 20; k++) {
            pis[k] /= sum_pi;
            if (pis[k] < 0.01f) pis[k] = 0.0f;
            sum2 += pis[k];
        }
        float r_pick = rng_uniform() * sum2;
        int chosen = 19;
        float cum = 0.0f;
        for (int k = 0; k < 20; k++) {
            cum += pis[k];
            if (r_pick <= cum) { chosen = k; break; }
        }

        float s1 = fast_expf(GMM_OUT[20 + chosen] - bias);
        float s2 = fast_expf(GMM_OUT[40 + chosen] - bias);
        if (s1 < 1e-4f) s1 = 1e-4f;
        if (s2 < 1e-4f) s2 = 1e-4f;
        float rho = fast_tanh(GMM_OUT[60 + chosen]);
        if (rho > 0.9999f) rho = 0.9999f;
        if (rho < -0.9999f) rho = -0.9999f;
        float mu1 = GMM_OUT[80 + chosen];
        float mu2 = GMM_OUT[100 + chosen];
        float es = fast_sigmoid(GMM_OUT[120]);
        if (es < 0.01f) es = 0.0f;

        float z1, z2;
        rng_normal2(&z1, &z2);
        float dx = mu1 + s1 * z1;
        float dy = mu2 + s2 * (rho * z1 + __builtin_sqrtf(1.0f - rho * rho + 1e-6f) * z2);
        float eos = (rng_uniform() < es) ? 1.0f : 0.0f;

        x0 = dx; x1 = dy; x2 = eos;
        OUT_BUF[out_n * 3 + 0] = dx;
        OUT_BUF[out_n * 3 + 1] = dy;
        OUT_BUF[out_n * 3 + 2] = eos;
        out_n++;

        if (best_u >= U - 2 && eos > 0.5f) break;
    }

    return out_n;
}
