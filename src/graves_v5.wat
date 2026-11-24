;; ============================================================================
;; Method C v5: Hand-Written WebAssembly Text (WAT) SIMD128 Graves Engine
;; Compiled with WABT:
;;   wat2wasm src/graves_v5.wat -o exports/handwriting/graves_v5.wasm
;;
;; Optimizations:
;;   1. Sparse char-context split: dense INT8 SIMD only over [x, h] (skips ~71/73 zero char cols!)
;;   2. 4-row unrolled i32x4.dot_i16x8_s + vectorized f32x4 dequant+bias store
;;   3. Vectorized f32x4 SIMD exp/sigmoid/tanh in lstm_update (4 lanes at once!)
;;   4. Tight primed states + streaming chunked API (init_session + step_chunk)
;; ============================================================================
(module
  ;; 128 pages = 8 MB linear memory
  (memory (export "memory") 128)

  ;; Memory layout constants (all 16-byte aligned)
  ;;   0x001000 H1 (1600 B)    0x001700 C1 (1600 B)
  ;;   0x001E00 H2 (1600 B)    0x002500 C2 (1600 B)
  ;;   0x002C00 H3 (1600 B)    0x003300 C3 (1600 B)
  ;;   0x003A00 KAPPA (40 B)   0x003B00 W_CTX (292 B)
  ;;   0x003D00 ACT_CHAR (73*8 B: [ch_i32, w_f32])
  ;;   0x004000 IN_F32 (3584 B) 0x005000 IN_I16 (1792 B)
  ;;   0x006000 GATES (6400 B)  0x008000 ATTN_OUT (128 B)
  ;;   0x008200 GMM_OUT (512 B) 0x008500 PIS (80 B)
  ;;   0x008600 ALPHA (40 B)    0x008640 INV_BETA (40 B)
  ;;   0x009000 CHARS_BUF (1024 B)
  ;;   0x00A000 OUT_BUF (24576 B)
  ;;   0x020000 WEIGHTS_BASE (graves_v5.bin loaded here by JS)

  ;; Weight tensor byte offsets (relative to WEIGHTS_BASE = 131072)
  (global $OFF_L1_DQ  (mut i32) (i32.const 0))
  (global $OFF_L1_DSC (mut i32) (i32.const 0))
  (global $OFF_L1_CQ  (mut i32) (i32.const 0))
  (global $OFF_L1_CSC (mut i32) (i32.const 0))
  (global $OFF_L1_B   (mut i32) (i32.const 0))

  (global $OFF_L2_DQ  (mut i32) (i32.const 0))
  (global $OFF_L2_DSC (mut i32) (i32.const 0))
  (global $OFF_L2_CQ  (mut i32) (i32.const 0))
  (global $OFF_L2_CSC (mut i32) (i32.const 0))
  (global $OFF_L2_B   (mut i32) (i32.const 0))

  (global $OFF_L3_DQ  (mut i32) (i32.const 0))
  (global $OFF_L3_DSC (mut i32) (i32.const 0))
  (global $OFF_L3_CQ  (mut i32) (i32.const 0))
  (global $OFF_L3_CSC (mut i32) (i32.const 0))
  (global $OFF_L3_B   (mut i32) (i32.const 0))

  (global $OFF_ATTN_WT (mut i32) (i32.const 0))
  (global $OFF_ATTN_B  (mut i32) (i32.const 0))
  (global $OFF_GMM_WT  (mut i32) (i32.const 0))
  (global $OFF_GMM_B   (mut i32) (i32.const 0))
  (global $OFF_PRIMED  (mut i32) (i32.const 0))

  ;; Active session state for chunked 60fps streaming
  (global $S_U         (mut i32) (i32.const 0))
  (global $S_STEP      (mut i32) (i32.const 0))
  (global $S_MAX_STEPS (mut i32) (i32.const 0))
  (global $S_DONE      (mut i32) (i32.const 1))
  (global $S_BIAS      (mut f32) (f32.const 1.05))
  (global $S_DK_SCALE  (mut f32) (f32.const 0.0424)) ;; 1.06 / 25.0
  (global $S_X0        (mut f32) (f32.const 0.0))
  (global $S_X1        (mut f32) (f32.const 0.0))
  (global $S_X2        (mut f32) (f32.const 1.0))
  (global $S_NACT      (mut i32) (i32.const 0))
  (global $RNG         (mut i32) (i32.const 42))
  (global $S_AT_END    (mut i32) (i32.const 0))
  (global $S_BACK_CNT  (mut i32) (i32.const 0))
  (global $S_CUR_X     (mut f32) (f32.const 0.0))
  (global $S_MAX_X     (mut f32) (f32.const -1e9))

  ;; Buffer pointer getters
  (func (export "get_weights_ptr") (result i32) (i32.const 131072))
  (func (export "get_chars_ptr")   (result i32) (i32.const 36864))  ;; 0x9000
  (func (export "get_out_ptr")     (result i32) (i32.const 40960))  ;; 0xA000

  ;; Configure tensor offsets from graves_v5_meta.json
  (func (export "init_offsets")
    (param $l1_dq i32) (param $l1_dsc i32) (param $l1_cq i32) (param $l1_csc i32) (param $l1_b i32)
    (param $l2_dq i32) (param $l2_dsc i32) (param $l2_cq i32) (param $l2_csc i32) (param $l2_b i32)
    (param $l3_dq i32) (param $l3_dsc i32) (param $l3_cq i32) (param $l3_csc i32) (param $l3_b i32)
    (param $attn_wt i32) (param $attn_b i32)
    (param $gmm_wt i32)  (param $gmm_b i32)
    (param $primed i32)
    (local $base i32)
    (local.set $base (i32.const 131072))
    (global.set $OFF_L1_DQ  (i32.add (local.get $base) (local.get $l1_dq)))
    (global.set $OFF_L1_DSC (i32.add (local.get $base) (local.get $l1_dsc)))
    (global.set $OFF_L1_CQ  (i32.add (local.get $base) (local.get $l1_cq)))
    (global.set $OFF_L1_CSC (i32.add (local.get $base) (local.get $l1_csc)))
    (global.set $OFF_L1_B   (i32.add (local.get $base) (local.get $l1_b)))

    (global.set $OFF_L2_DQ  (i32.add (local.get $base) (local.get $l2_dq)))
    (global.set $OFF_L2_DSC (i32.add (local.get $base) (local.get $l2_dsc)))
    (global.set $OFF_L2_CQ  (i32.add (local.get $base) (local.get $l2_cq)))
    (global.set $OFF_L2_CSC (i32.add (local.get $base) (local.get $l2_csc)))
    (global.set $OFF_L2_B   (i32.add (local.get $base) (local.get $l2_b)))

    (global.set $OFF_L3_DQ  (i32.add (local.get $base) (local.get $l3_dq)))
    (global.set $OFF_L3_DSC (i32.add (local.get $base) (local.get $l3_dsc)))
    (global.set $OFF_L3_CQ  (i32.add (local.get $base) (local.get $l3_cq)))
    (global.set $OFF_L3_CSC (i32.add (local.get $base) (local.get $l3_csc)))
    (global.set $OFF_L3_B   (i32.add (local.get $base) (local.get $l3_b)))

    (global.set $OFF_ATTN_WT (i32.add (local.get $base) (local.get $attn_wt)))
    (global.set $OFF_ATTN_B  (i32.add (local.get $base) (local.get $attn_b)))
    (global.set $OFF_GMM_WT  (i32.add (local.get $base) (local.get $gmm_wt)))
    (global.set $OFF_GMM_B   (i32.add (local.get $base) (local.get $gmm_b)))
    (global.set $OFF_PRIMED  (i32.add (local.get $base) (local.get $primed)))
  )

  ;; ==========================================================================
  ;; SIMD & Scalar Fast Math Primitives
  ;; ==========================================================================
  ;; Vectorized 4-lane exp(x) for x clamped to [-20, 20]
  (func $exp_v4 (param $x v128) (result v128)
    (local $fx v128) (local $n v128) (local $fn v128) (local $r v128) (local $p v128) (local $pow2 v128)
    ;; clamp x to [-20, 20]
    (local.set $x (f32x4.min (f32x4.max (local.get $x) (v128.const f32x4 -20.0 -20.0 -20.0 -20.0))
                             (v128.const f32x4 20.0 20.0 20.0 20.0)))
    (local.set $fx (f32x4.mul (local.get $x) (v128.const f32x4 1.44269504 1.44269504 1.44269504 1.44269504)))
    (local.set $fn (f32x4.nearest (local.get $fx)))
    (local.set $n  (i32x4.trunc_sat_f32x4_s (local.get $fn)))
    ;; r = x - fn * ln2
    (local.set $r  (f32x4.sub (local.get $x) (f32x4.mul (local.get $fn) (v128.const f32x4 0.69314718 0.69314718 0.69314718 0.69314718))))
    ;; Horner degree-4 polynomial for e^r
    (local.set $p (f32x4.add (v128.const f32x4 0.16666667 0.16666667 0.16666667 0.16666667)
                             (f32x4.mul (local.get $r) (v128.const f32x4 0.04166667 0.04166667 0.04166667 0.04166667))))
    (local.set $p (f32x4.add (v128.const f32x4 0.5 0.5 0.5 0.5) (f32x4.mul (local.get $r) (local.get $p))))
    (local.set $p (f32x4.add (v128.const f32x4 1.0 1.0 1.0 1.0) (f32x4.mul (local.get $r) (local.get $p))))
    (local.set $p (f32x4.add (v128.const f32x4 1.0 1.0 1.0 1.0) (f32x4.mul (local.get $r) (local.get $p))))
    ;; pow2 = (n + 127) << 23
    (local.set $pow2 (i32x4.shl (i32x4.add (local.get $n) (v128.const i32x4 127 127 127 127)) (i32.const 23)))
    (f32x4.mul (local.get $p) (local.get $pow2))
  )

  ;; Vectorized 4-lane sigmoid(x) = 1 / (1 + exp(-x))
  (func $sigmoid_v4 (param $x v128) (result v128)
    (f32x4.div (v128.const f32x4 1.0 1.0 1.0 1.0)
               (f32x4.add (v128.const f32x4 1.0 1.0 1.0 1.0) (call $exp_v4 (f32x4.neg (local.get $x)))))
  )

  ;; Vectorized 4-lane tanh(x) = 2 * sigmoid(2x) - 1
  (func $tanh_v4 (param $x v128) (result v128)
    (f32x4.sub
      (f32x4.mul (v128.const f32x4 2.0 2.0 2.0 2.0)
                 (call $sigmoid_v4 (f32x4.mul (local.get $x) (v128.const f32x4 2.0 2.0 2.0 2.0))))
      (v128.const f32x4 1.0 1.0 1.0 1.0))
  )

  ;; Scalar fast_exp(x)
  (func $expf (param $x f32) (result f32)
    (local $fn f32) (local $n i32) (local $r f32) (local $p f32)
    (if (f32.lt (local.get $x) (f32.const -80.0)) (then (return (f32.const 0.0))))
    (if (f32.gt (local.get $x) (f32.const 80.0))  (then (local.set $x (f32.const 80.0))))
    (local.set $fn (f32.nearest (f32.mul (local.get $x) (f32.const 1.44269504))))
    (local.set $n  (i32.trunc_sat_f32_s (local.get $fn)))
    (local.set $r  (f32.sub (local.get $x) (f32.mul (local.get $fn) (f32.const 0.69314718))))
    (local.set $p  (f32.add (f32.const 0.16666667) (f32.mul (local.get $r) (f32.const 0.04166667))))
    (local.set $p  (f32.add (f32.const 0.5) (f32.mul (local.get $r) (local.get $p))))
    (local.set $p  (f32.add (f32.const 1.0) (f32.mul (local.get $r) (local.get $p))))
    (local.set $p  (f32.add (f32.const 1.0) (f32.mul (local.get $r) (local.get $p))))
    (f32.mul (local.get $p) (f32.reinterpret_i32 (i32.shl (i32.add (local.get $n) (i32.const 127)) (i32.const 23))))
  )

  ;; Scalar fast_log(x)
  (func $logf (param $x f32) (result f32)
    (local $u i32) (local $e i32) (local $m f32) (local $y f32) (local $y2 f32)
    (if (f32.le (local.get $x) (f32.const 1e-30)) (then (return (f32.const -69.07755))))
    (local.set $u (i32.reinterpret_f32 (local.get $x)))
    (local.set $e (i32.sub (i32.and (i32.shr_u (local.get $u) (i32.const 23)) (i32.const 255)) (i32.const 127)))
    (local.set $m (f32.reinterpret_i32 (i32.or (i32.and (local.get $u) (i32.const 0x007FFFFF)) (i32.const 0x3F800000))))
    (local.set $y (f32.div (f32.sub (local.get $m) (f32.const 1.0)) (f32.add (local.get $m) (f32.const 1.0))))
    (local.set $y2 (f32.mul (local.get $y) (local.get $y)))
    (f32.add
      (f32.mul (f32.mul (f32.const 2.0) (local.get $y))
        (f32.add (f32.const 1.0)
          (f32.mul (local.get $y2)
            (f32.add (f32.const 0.33333333)
              (f32.mul (local.get $y2) (f32.add (f32.const 0.2) (f32.mul (local.get $y2) (f32.const 0.14285714))))))))
      (f32.mul (f32.convert_i32_s (local.get $e)) (f32.const 0.69314718)))
  )

  (func $softplus (param $x f32) (result f32)
    (if (f32.gt (local.get $x) (f32.const 20.0))  (then (return (local.get $x))))
    (if (f32.lt (local.get $x) (f32.const -20.0)) (then (return (f32.const 0.0))))
    (call $logf (f32.add (f32.const 1.0) (call $expf (local.get $x))))
  )

  (func $sigmoid (param $x f32) (result f32)
    (if (f32.lt (local.get $x) (f32.const -20.0)) (then (return (f32.const 0.0))))
    (if (f32.gt (local.get $x) (f32.const 20.0))  (then (return (f32.const 1.0))))
    (f32.div (f32.const 1.0) (f32.add (f32.const 1.0) (call $expf (f32.neg (local.get $x)))))
  )

  (func $tanhf (param $x f32) (result f32)
    (f32.sub (f32.mul (f32.const 2.0) (call $sigmoid (f32.mul (local.get $x) (f32.const 2.0)))) (f32.const 1.0))
  )

  ;; Fast xorshift32 uniform [0, 1)
  (func $rng_u (result f32)
    (local $x i32)
    (local.set $x (global.get $RNG))
    (local.set $x (i32.xor (local.get $x) (i32.shl (local.get $x) (i32.const 13))))
    (local.set $x (i32.xor (local.get $x) (i32.shr_u (local.get $x) (i32.const 17))))
    (local.set $x (i32.xor (local.get $x) (i32.shl (local.get $x) (i32.const 5))))
    (global.set $RNG (local.get $x))
    (f32.mul (f32.convert_i32_u (i32.and (local.get $x) (i32.const 0x00FFFFFF))) (f32.const 5.9604644775e-8))
  )

  (func $cosf (param $x f32) (result f32)
    (local $k i32) (local $sign f32) (local $x2 f32)
    (local.set $k (i32.trunc_sat_f32_s (f32.div (local.get $x) (f32.const 6.2831853))))
    (local.set $x (f32.sub (local.get $x) (f32.mul (f32.convert_i32_s (local.get $k)) (f32.const 6.2831853))))
    (if (f32.gt (local.get $x) (f32.const 3.14159265))  (then (local.set $x (f32.sub (local.get $x) (f32.const 6.2831853)))))
    (if (f32.lt (local.get $x) (f32.const -3.14159265)) (then (local.set $x (f32.add (local.get $x) (f32.const 6.2831853)))))
    (local.set $sign (f32.const 1.0))
    (if (f32.gt (local.get $x) (f32.const 1.5707963))
      (then (local.set $x (f32.sub (f32.const 3.14159265) (local.get $x))) (local.set $sign (f32.const -1.0)))
      (else (if (f32.lt (local.get $x) (f32.const -1.5707963))
        (then (local.set $x (f32.sub (f32.const -3.14159265) (local.get $x))) (local.set $sign (f32.const -1.0))))))
    (local.set $x2 (f32.mul (local.get $x) (local.get $x)))
    (f32.mul (local.get $sign)
      (f32.add (f32.const 1.0)
        (f32.mul (local.get $x2)
          (f32.add (f32.const -0.5)
            (f32.mul (local.get $x2) (f32.sub (f32.const 0.04166667) (f32.mul (local.get $x2) (f32.const 0.00138889))))))))
  )

  ;; Horizontal sum of 4 i32 lanes
  (func $hsum_i32 (param $v v128) (result i32)
    (i32.add
      (i32.add (i32x4.extract_lane 0 (local.get $v)) (i32x4.extract_lane 1 (local.get $v)))
      (i32.add (i32x4.extract_lane 2 (local.get $v)) (i32x4.extract_lane 3 (local.get $v))))
  )

  ;; Horizontal sum of 4 f32 lanes
  (func $hsum_f32 (param $v v128) (result f32)
    (f32.add
      (f32.add (f32x4.extract_lane 0 (local.get $v)) (f32x4.extract_lane 1 (local.get $v)))
      (f32.add (f32x4.extract_lane 2 (local.get $v)) (f32x4.extract_lane 3 (local.get $v))))
  )

  ;; ==========================================================================
  ;; HOT KERNEL 1: 4-Row Unrolled INT8 SIMD128 Dense Matvec + Vectorized Dequant
  ;;   Reads IN_F32[0..in_dim-1], quantizes once into IN_I16[0..in_pad-1],
  ;;   writes 1600 outputs into GATES (0x6000) = dense_dot * x_sc * dsc + bias
  ;; ==========================================================================
  (func $matvec_dense_i8
    (param $in_dim i32) (param $in_pad i32)
    (param $dq_ptr i32) (param $dsc_ptr i32) (param $b_ptr i32)
    (local $i i32) (local $end i32)
    (local $vmax v128) (local $max_abs f32) (local $x_scale f32) (local $inv_sc f32) (local $vinv v128)
    (local $v0 v128) (local $v1 v128)
    (local $r i32) (local $row_stride i32)
    (local $p0 i32) (local $p1 i32) (local $p2 i32) (local $p3 i32)
    (local $xp i32) (local $xend i32)
    (local $acc0 v128) (local $acc1 v128) (local $acc2 v128) (local $acc3 v128)
    (local $xlo v128) (local $xhi v128) (local $w v128)
    (local $vxsc v128) (local $sums v128) (local $r_off i32)

    ;; Zero padding tail of IN_F32[in_dim..in_pad-1]
    (local.set $i (local.get $in_dim))
    (loop $zpad
      (if (i32.lt_u (local.get $i) (local.get $in_pad))
        (then
          (f32.store (i32.add (i32.const 16384) (i32.shl (local.get $i) (i32.const 2))) (f32.const 0.0))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $zpad))))

    ;; 1. Vectorized max_abs over IN_F32 (0x4000 = 16384)
    (local.set $vmax (v128.const f32x4 0.0 0.0 0.0 0.0))
    (local.set $i (i32.const 16384))
    (local.set $end (i32.add (i32.const 16384) (i32.shl (local.get $in_pad) (i32.const 2))))
    (loop $max_loop
      (local.set $vmax (f32x4.max (local.get $vmax) (f32x4.abs (v128.load (local.get $i)))))
      (local.set $i (i32.add (local.get $i) (i32.const 16)))
      (br_if $max_loop (i32.lt_u (local.get $i) (local.get $end))))

    (local.set $max_abs
      (f32.max
        (f32.max (f32x4.extract_lane 0 (local.get $vmax)) (f32x4.extract_lane 1 (local.get $vmax)))
        (f32.max (f32x4.extract_lane 2 (local.get $vmax)) (f32x4.extract_lane 3 (local.get $vmax)))))
    (local.set $x_scale (select (f32.mul (local.get $max_abs) (f32.const 0.0078740157)) (f32.const 1.0)
                                (f32.gt (local.get $max_abs) (f32.const 1e-8))))
    (local.set $inv_sc (f32.div (f32.const 1.0) (local.get $x_scale)))
    (local.set $vinv (f32x4.splat (local.get $inv_sc)))

    ;; 2. Vectorized quantize IN_F32 -> IN_I16 (0x5000 = 20480), 8 floats -> 8 int16 per iter!
    (local.set $i (i32.const 16384))
    (local.set $xp (i32.const 20480))
    (loop $q_loop
      (local.set $v0 (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (v128.load (local.get $i)) (local.get $vinv)))))
      (local.set $v1 (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (v128.load offset=16 (local.get $i)) (local.get $vinv)))))
      (v128.store (local.get $xp) (i16x8.narrow_i32x4_s (local.get $v0) (local.get $v1)))
      (local.set $i  (i32.add (local.get $i)  (i32.const 32)))
      (local.set $xp (i32.add (local.get $xp) (i32.const 16)))
      (br_if $q_loop (i32.lt_u (local.get $i) (local.get $end))))

    ;; 3. 4-row unrolled INT8xINT16 SIMD dot product over 1600 rows
    (local.set $vxsc (f32x4.splat (local.get $x_scale)))
    (local.set $row_stride (local.get $in_pad))
    (local.set $xend (i32.add (i32.const 20480) (i32.shl (local.get $in_pad) (i32.const 1))))
    (local.set $p0 (local.get $dq_ptr))
    (local.set $r (i32.const 0))

    (loop $row_loop
      (local.set $p1 (i32.add (local.get $p0) (local.get $row_stride)))
      (local.set $p2 (i32.add (local.get $p1) (local.get $row_stride)))
      (local.set $p3 (i32.add (local.get $p2) (local.get $row_stride)))

      (local.set $acc0 (v128.const i32x4 0 0 0 0))
      (local.set $acc1 (v128.const i32x4 0 0 0 0))
      (local.set $acc2 (v128.const i32x4 0 0 0 0))
      (local.set $acc3 (v128.const i32x4 0 0 0 0))
      (local.set $xp (i32.const 20480))

      (loop $col_loop
        ;; Load 16 int16 activations ONCE for all 4 rows (hot in L1!)
        (local.set $xlo (v128.load (local.get $xp)))
        (local.set $xhi (v128.load offset=16 (local.get $xp)))

        ;; Row 0
        (local.set $w (v128.load (local.get $p0)))
        (local.set $acc0 (i32x4.add (local.get $acc0) (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s  (local.get $w)) (local.get $xlo))))
        (local.set $acc0 (i32x4.add (local.get $acc0) (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $w)) (local.get $xhi))))

        ;; Row 1
        (local.set $w (v128.load (local.get $p1)))
        (local.set $acc1 (i32x4.add (local.get $acc1) (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s  (local.get $w)) (local.get $xlo))))
        (local.set $acc1 (i32x4.add (local.get $acc1) (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $w)) (local.get $xhi))))

        ;; Row 2
        (local.set $w (v128.load (local.get $p2)))
        (local.set $acc2 (i32x4.add (local.get $acc2) (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s  (local.get $w)) (local.get $xlo))))
        (local.set $acc2 (i32x4.add (local.get $acc2) (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $w)) (local.get $xhi))))

        ;; Row 3
        (local.set $w (v128.load (local.get $p3)))
        (local.set $acc3 (i32x4.add (local.get $acc3) (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s  (local.get $w)) (local.get $xlo))))
        (local.set $acc3 (i32x4.add (local.get $acc3) (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $w)) (local.get $xhi))))

        (local.set $p0 (i32.add (local.get $p0) (i32.const 16)))
        (local.set $p1 (i32.add (local.get $p1) (i32.const 16)))
        (local.set $p2 (i32.add (local.get $p2) (i32.const 16)))
        (local.set $p3 (i32.add (local.get $p3) (i32.const 16)))
        (local.set $xp (i32.add (local.get $xp) (i32.const 32)))
        (br_if $col_loop (i32.lt_u (local.get $xp) (local.get $xend))))

      ;; Pack 4 horizontal sums -> f32x4 -> scale + bias -> store 4 gates!
      (local.set $sums (i32x4.replace_lane 0 (v128.const i32x4 0 0 0 0) (call $hsum_i32 (local.get $acc0))))
      (local.set $sums (i32x4.replace_lane 1 (local.get $sums) (call $hsum_i32 (local.get $acc1))))
      (local.set $sums (i32x4.replace_lane 2 (local.get $sums) (call $hsum_i32 (local.get $acc2))))
      (local.set $sums (i32x4.replace_lane 3 (local.get $sums) (call $hsum_i32 (local.get $acc3))))

      (local.set $r_off (i32.shl (local.get $r) (i32.const 2)))
      (v128.store (i32.add (i32.const 24576) (local.get $r_off)) ;; GATES = 0x6000 = 24576
        (f32x4.add
          (f32x4.mul
            (f32x4.convert_i32x4_s (local.get $sums))
            (f32x4.mul (v128.load (i32.add (local.get $dsc_ptr) (local.get $r_off))) (local.get $vxsc)))
          (v128.load (i32.add (local.get $b_ptr) (local.get $r_off)))))

      ;; Advance p0 past rows 1,2,3 (p3 is already at end of row 3!)
      (local.set $p0 (local.get $p3))
      (local.set $r (i32.add (local.get $r) (i32.const 4)))
      (br_if $row_loop (i32.lt_u (local.get $r) (i32.const 1600))))
  )

  ;; ==========================================================================
  ;; HOT KERNEL 2: Sparse Character Column SIMD Add into GATES[0..1599]
  ;;   Only runs for the ~2 active chars where w[ch] > 1e-3 (skips ~97%!)
  ;; ==========================================================================
  (func $add_sparse_chars (param $cq_ptr i32) (param $csc_ptr i32)
    (local $a i32) (local $nact i32) (local $aptr i32)
    (local $ch i32) (local $wval f32) (local $sc f32) (local $vsc v128)
    (local $cp i32) (local $gp i32) (local $gend i32)
    (local $b16 v128) (local $lo8 v128) (local $hi8 v128)

    (local.set $nact (global.get $S_NACT))
    (if (i32.eqz (local.get $nact)) (then (return)))
    (local.set $a (i32.const 0))
    (local.set $gend (i32.const 30976)) ;; 24576 + 1600*4

    (loop $act_loop
      (local.set $aptr (i32.add (i32.const 15616) (i32.shl (local.get $a) (i32.const 3)))) ;; ACT_CHAR = 0x3D00 = 15616
      (local.set $ch   (i32.load (local.get $aptr)))
      (local.set $wval (f32.load offset=4 (local.get $aptr)))
      (local.set $sc   (f32.mul (local.get $wval) (f32.load (i32.add (local.get $csc_ptr) (i32.shl (local.get $ch) (i32.const 2))))))
      (local.set $vsc  (f32x4.splat (local.get $sc)))
      (local.set $cp   (i32.add (local.get $cq_ptr) (i32.mul (local.get $ch) (i32.const 1600))))
      (local.set $gp   (i32.const 24576))

      (loop $g_loop
        ;; Unpack 16 int8 weights -> 4 x f32x4 and FMA into GATES
        (local.set $b16 (v128.load (local.get $cp)))
        (local.set $lo8 (i16x8.extend_low_i8x16_s  (local.get $b16)))
        (local.set $hi8 (i16x8.extend_high_i8x16_s (local.get $b16)))

        (v128.store (local.get $gp)
          (f32x4.add (v128.load (local.get $gp))
            (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $lo8))) (local.get $vsc))))
        (v128.store offset=16 (local.get $gp)
          (f32x4.add (v128.load offset=16 (local.get $gp))
            (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $lo8))) (local.get $vsc))))
        (v128.store offset=32 (local.get $gp)
          (f32x4.add (v128.load offset=32 (local.get $gp))
            (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $hi8))) (local.get $vsc))))
        (v128.store offset=48 (local.get $gp)
          (f32x4.add (v128.load offset=48 (local.get $gp))
            (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $hi8))) (local.get $vsc))))

        (local.set $cp (i32.add (local.get $cp) (i32.const 16)))
        (local.set $gp (i32.add (local.get $gp) (i32.const 64)))
        (br_if $g_loop (i32.lt_u (local.get $gp) (local.get $gend))))

      (local.set $a (i32.add (local.get $a) (i32.const 1)))
      (br_if $act_loop (i32.lt_u (local.get $a) (local.get $nact))))
  )

  ;; ==========================================================================
  ;; HOT KERNEL 3: Vectorized f32x4 LSTM Gate Update (4 hidden units per iter!)
  ;; ==========================================================================
  (func $lstm_update_simd (param $h_ptr i32) (param $c_ptr i32)
    (local $off i32) (local $gi v128) (local $gj v128) (local $gf v128) (local $go v128) (local $cn v128)
    (local.set $off (i32.const 0))
    (loop $u_loop
      ;; GATES[0..399]=i, [400..799]=j (+1600 B), [800..1199]=f (+3200 B), [1200..1599]=o (+4800 B)
      (local.set $gi (call $sigmoid_v4 (v128.load (i32.add (i32.const 24576) (local.get $off)))))
      (local.set $gj (call $tanh_v4    (v128.load (i32.add (i32.const 26176) (local.get $off)))))
      (local.set $gf (call $sigmoid_v4 (f32x4.add (v128.load (i32.add (i32.const 27776) (local.get $off)))
                                                  (v128.const f32x4 1.0 1.0 1.0 1.0))))
      (local.set $go (call $sigmoid_v4 (v128.load (i32.add (i32.const 29376) (local.get $off)))))
      (local.set $cn (f32x4.add (f32x4.mul (v128.load (i32.add (local.get $c_ptr) (local.get $off))) (local.get $gf))
                                (f32x4.mul (local.get $gi) (local.get $gj))))
      (v128.store (i32.add (local.get $c_ptr) (local.get $off)) (local.get $cn))
      (v128.store (i32.add (local.get $h_ptr) (local.get $off)) (f32x4.mul (call $tanh_v4 (local.get $cn)) (local.get $go)))
      (local.set $off (i32.add (local.get $off) (i32.const 16)))
      (br_if $u_loop (i32.lt_u (local.get $off) (i32.const 1600))))
  )

  ;; Small FP32 SIMD128 matvec (for attn 30x480 and gmm 121x400)
  (func $matvec_f32_simd (param $x_ptr i32) (param $in_pad i32) (param $out_dim i32) (param $wt_ptr i32) (param $b_ptr i32) (param $out_ptr i32)
    (local $r i32) (local $wp i32) (local $xp i32) (local $xend i32) (local $acc v128)
    (local.set $xend (i32.add (local.get $x_ptr) (i32.shl (local.get $in_pad) (i32.const 2))))
    (local.set $wp (local.get $wt_ptr))
    (local.set $r (i32.const 0))
    (loop $rl
      (local.set $acc (v128.const f32x4 0.0 0.0 0.0 0.0))
      (local.set $xp (local.get $x_ptr))
      (loop $cl
        (local.set $acc (f32x4.add (local.get $acc) (f32x4.mul (v128.load (local.get $wp)) (v128.load (local.get $xp)))))
        (local.set $wp (i32.add (local.get $wp) (i32.const 16)))
        (local.set $xp (i32.add (local.get $xp) (i32.const 16)))
        (br_if $cl (i32.lt_u (local.get $xp) (local.get $xend))))
      (f32.store (i32.add (local.get $out_ptr) (i32.shl (local.get $r) (i32.const 2)))
        (f32.add (call $hsum_f32 (local.get $acc)) (f32.load (i32.add (local.get $b_ptr) (i32.shl (local.get $r) (i32.const 2))))))
      (local.set $r (i32.add (local.get $r) (i32.const 1)))
      (br_if $rl (i32.lt_u (local.get $r) (local.get $out_dim))))
  )

  ;; Helper: rebuild sparse ACT_CHAR list from W_CTX[0..72]
  (func $rebuild_act_chars
    (local $c i32) (local $wv f32) (local $n i32) (local $ap i32)
    (local.set $n (i32.const 0))
    (local.set $c (i32.const 0))
    (loop $cl
      (local.set $wv (f32.load (i32.add (i32.const 15104) (i32.shl (local.get $c) (i32.const 2))))) ;; W_CTX = 0x3B00 = 15104
      (if (f32.gt (local.get $wv) (f32.const 1e-3))
        (then
          (local.set $ap (i32.add (i32.const 15616) (i32.shl (local.get $n) (i32.const 3))))
          (i32.store (local.get $ap) (local.get $c))
          (f32.store offset=4 (local.get $ap) (local.get $wv))
          (local.set $n (i32.add (local.get $n) (i32.const 1)))))
      (local.set $c (i32.add (local.get $c) (i32.const 1)))
      (br_if $cl (i32.lt_u (local.get $c) (i32.const 73))))
    (global.set $S_NACT (local.get $n))
  )

  ;; Helper: copy n floats (aligned or unaligned via bulk memory.copy!)
  ;; Initialize a synthesis session (loads tight primed state for style_id)
  (func (export "init_session")
    (param $style_id i32) (param $U i32) (param $bias f32) (param $dk_mult f32) (param $max_steps i32) (param $seed i32)
    (local $ps i32)
    (if (i32.or (i32.lt_s (local.get $style_id) (i32.const 0)) (i32.gt_s (local.get $style_id) (i32.const 12)))
      (then (local.set $style_id (i32.const 9))))
    (local.set $ps (i32.add (global.get $OFF_PRIMED) (i32.mul (local.get $style_id) (i32.const 9932)))) ;; 2483 * 4

    ;; Copy primed state vectors into scratch memory
    (memory.copy (i32.const 4096)  (local.get $ps) (i32.const 1600))                           ;; H1
    (memory.copy (i32.const 5888)  (i32.add (local.get $ps) (i32.const 1600)) (i32.const 1600));; C1
    (memory.copy (i32.const 7680)  (i32.add (local.get $ps) (i32.const 3200)) (i32.const 1600));; H2
    (memory.copy (i32.const 9472)  (i32.add (local.get $ps) (i32.const 4800)) (i32.const 1600));; C2
    (memory.copy (i32.const 11264) (i32.add (local.get $ps) (i32.const 6400)) (i32.const 1600));; H3
    (memory.copy (i32.const 13056) (i32.add (local.get $ps) (i32.const 8000)) (i32.const 1600));; C3
    (memory.copy (i32.const 14848) (i32.add (local.get $ps) (i32.const 9600)) (i32.const 40))  ;; KAPPA
    (memory.copy (i32.const 15104) (i32.add (local.get $ps) (i32.const 9640)) (i32.const 292)) ;; W_CTX

    (call $rebuild_act_chars)
    (global.set $S_U (local.get $U))
    (global.set $S_STEP (i32.const 0))
    (global.set $S_MAX_STEPS (select (local.get $max_steps) (i32.const 2048) (i32.le_s (local.get $max_steps) (i32.const 2048))))
    (global.set $S_DONE (i32.const 0))
    (global.set $S_BIAS (local.get $bias))
    (global.set $S_DK_SCALE (f32.mul (local.get $dk_mult) (f32.const 0.04))) ;; dk_mult / 25.0
    (global.set $S_X0 (f32.const 0.0))
    (global.set $S_X1 (f32.const 0.0))
    (global.set $S_X2 (f32.const 1.0))
    (global.set $RNG (select (local.get $seed) (i32.const 42) (i32.ne (local.get $seed) (i32.const 0))))
    (global.set $S_AT_END (i32.const 0))
    (global.set $S_BACK_CNT (i32.const 0))
    (global.set $S_CUR_X (f32.const 0.0))
    (global.set $S_MAX_X (f32.const -1e9))
  )

  ;; ==========================================================================
  ;; Run up to $chunk_steps steps; writes [dx, dy, eos] to OUT_BUF (0xA000).
  ;; Returns:
  ;;   +n : produced n points, MORE steps remain (stream this chunk immediately!)
  ;;   -n : produced n points, FINISHED cleanly!
  ;; ==========================================================================
  (func (export "step_chunk") (param $chunk_steps i32) (result i32)
    (local $out_n i32) (local $k i32) (local $u i32) (local $U i32)
    (local $b f32) (local $fu f32) (local $s f32) (local $d f32) (local $arg f32)
    (local $best_u i32) (local $best_phi f32) (local $ch i32) (local $wptr i32)
    (local $bias f32) (local $b1 f32) (local $max_pi f32) (local $sum_pi f32) (local $sum2 f32)
    (local $r_pick f32) (local $cum f32) (local $chosen i32)
    (local $s1 f32) (local $s2 f32) (local $rho f32) (local $mu1 f32) (local $mu2 f32) (local $es f32)
    (local $u1 f32) (local $u2 f32) (local $rad f32) (local $th f32) (local $z1 f32) (local $z2 f32)
    (local $dx f32) (local $dy f32) (local $eos f32) (local $op i32)

    (if (global.get $S_DONE) (then (return (i32.const 0))))
    (local.set $U (global.get $S_U))
    (local.set $bias (global.get $S_BIAS))
    (local.set $b1 (f32.add (f32.const 1.0) (local.get $bias)))
    (local.set $out_n (i32.const 0))

    (block $done_blk
      (loop $step_loop
        (br_if $done_blk (i32.ge_s (local.get $out_n) (local.get $chunk_steps)))
        (if (i32.ge_s (global.get $S_STEP) (global.get $S_MAX_STEPS))
          (then (global.set $S_DONE (i32.const 1)) (br $done_blk)))

        ;; ---- 1. LSTM 1: dense [x(3), h1(400)] = 403 (pad 416) + sparse w_ctx ----
        (f32.store (i32.const 16384) (global.get $S_X0))
        (f32.store (i32.const 16388) (global.get $S_X1))
        (f32.store (i32.const 16392) (global.get $S_X2))
        (memory.copy (i32.const 16396) (i32.const 4096) (i32.const 1600))
        (call $matvec_dense_i8 (i32.const 403) (i32.const 416)
              (global.get $OFF_L1_DQ) (global.get $OFF_L1_DSC) (global.get $OFF_L1_B))
        (call $add_sparse_chars (global.get $OFF_L1_CQ) (global.get $OFF_L1_CSC))
        (call $lstm_update_simd (i32.const 4096) (i32.const 5888))

        ;; ---- 2. Attention: [w(73), x(3), new_h1(400)] = 476 (pad 480) ----
        (memory.copy (i32.const 16384) (i32.const 15104) (i32.const 292))
        (f32.store (i32.const 16676) (global.get $S_X0))
        (f32.store (i32.const 16680) (global.get $S_X1))
        (f32.store (i32.const 16684) (global.get $S_X2))
        (memory.copy (i32.const 16688) (i32.const 4096) (i32.const 1600))
        (v128.store (i32.const 18288) (v128.const f32x4 0.0 0.0 0.0 0.0)) ;; pad 476..479
        (call $matvec_f32_simd (i32.const 16384) (i32.const 480) (i32.const 30)
              (global.get $OFF_ATTN_WT) (global.get $OFF_ATTN_B) (i32.const 32768))

        ;; Update alpha, inv_beta, kappa
        (local.set $k (i32.const 0))
        (loop $ak
          (f32.store (i32.add (i32.const 34304) (i32.shl (local.get $k) (i32.const 2)))
            (call $softplus (f32.load (i32.add (i32.const 32768) (i32.shl (local.get $k) (i32.const 2))))))
          (local.set $b (call $softplus (f32.load (i32.add (i32.const 32808) (i32.shl (local.get $k) (i32.const 2))))))
          (if (f32.lt (local.get $b) (f32.const 0.01)) (then (local.set $b (f32.const 0.01))))
          (f32.store (i32.add (i32.const 34368) (i32.shl (local.get $k) (i32.const 2))) (f32.div (f32.const 1.0) (local.get $b)))
          (f32.store (i32.add (i32.const 14848) (i32.shl (local.get $k) (i32.const 2)))
            (f32.add
              (f32.load (i32.add (i32.const 14848) (i32.shl (local.get $k) (i32.const 2))))
              (f32.mul (call $softplus (f32.load (i32.add (i32.const 32848) (i32.shl (local.get $k) (i32.const 2)))))
                       (global.get $S_DK_SCALE))))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br_if $ak (i32.lt_u (local.get $k) (i32.const 10))))

        ;; Compute phi[u] -> sparse W_CTX
        (memory.fill (i32.const 15104) (i32.const 0) (i32.const 292))
        (local.set $best_u (i32.const 0))
        (local.set $best_phi (f32.const -1.0))
        (local.set $u (i32.const 0))
        (loop $ul
          (local.set $fu (f32.convert_i32_s (local.get $u)))
          (local.set $s (f32.const 0.0))
          (local.set $k (i32.const 0))
          (loop $kl
            (local.set $d (f32.sub (f32.load (i32.add (i32.const 14848) (i32.shl (local.get $k) (i32.const 2)))) (local.get $fu)))
            (local.set $arg (f32.mul (f32.neg (f32.mul (local.get $d) (local.get $d)))
                                     (f32.load (i32.add (i32.const 34368) (i32.shl (local.get $k) (i32.const 2))))))
            (if (f32.gt (local.get $arg) (f32.const -25.0))
              (then (local.set $s (f32.add (local.get $s)
                (f32.mul (f32.load (i32.add (i32.const 34304) (i32.shl (local.get $k) (i32.const 2))))
                         (call $expf (local.get $arg)))))))
            (local.set $k (i32.add (local.get $k) (i32.const 1)))
            (br_if $kl (i32.lt_u (local.get $k) (i32.const 10))))
          (if (f32.gt (local.get $s) (local.get $best_phi))
            (then (local.set $best_phi (local.get $s)) (local.set $best_u (local.get $u))))
          (local.set $ch (i32.load (i32.add (i32.const 36864) (i32.shl (local.get $u) (i32.const 2)))))
          (if (i32.lt_u (local.get $ch) (i32.const 73))
            (then
              (local.set $wptr (i32.add (i32.const 15104) (i32.shl (local.get $ch) (i32.const 2))))
              (f32.store (local.get $wptr) (f32.add (f32.load (local.get $wptr)) (local.get $s)))))
          (local.set $u (i32.add (local.get $u) (i32.const 1)))
          (br_if $ul (i32.lt_s (local.get $u) (local.get $U))))
        (call $rebuild_act_chars)

        ;; ---- 3. LSTM 2: dense [x(3), h1(400), h2(400)] = 803 (pad 816) + sparse w_ctx ----
        (f32.store (i32.const 16384) (global.get $S_X0))
        (f32.store (i32.const 16388) (global.get $S_X1))
        (f32.store (i32.const 16392) (global.get $S_X2))
        (memory.copy (i32.const 16396) (i32.const 4096) (i32.const 1600))
        (memory.copy (i32.const 17996) (i32.const 7680) (i32.const 1600))
        (call $matvec_dense_i8 (i32.const 803) (i32.const 816)
              (global.get $OFF_L2_DQ) (global.get $OFF_L2_DSC) (global.get $OFF_L2_B))
        (call $add_sparse_chars (global.get $OFF_L2_CQ) (global.get $OFF_L2_CSC))
        (call $lstm_update_simd (i32.const 7680) (i32.const 9472))

        ;; ---- 4. LSTM 3: dense [x(3), h2(400), h3(400)] = 803 (pad 816) + sparse w_ctx ----
        (memory.copy (i32.const 16396) (i32.const 7680)  (i32.const 1600))
        (memory.copy (i32.const 17996) (i32.const 11264) (i32.const 1600))
        (call $matvec_dense_i8 (i32.const 803) (i32.const 816)
              (global.get $OFF_L3_DQ) (global.get $OFF_L3_DSC) (global.get $OFF_L3_B))
        (call $add_sparse_chars (global.get $OFF_L3_CQ) (global.get $OFF_L3_CSC))
        (call $lstm_update_simd (i32.const 11264) (i32.const 13056))

        ;; ---- 5. GMM head: h3(400) -> 121 (0x8200 = 33280) ----
        (call $matvec_f32_simd (i32.const 11264) (i32.const 400) (i32.const 121)
              (global.get $OFF_GMM_WT) (global.get $OFF_GMM_B) (i32.const 33280))

        ;; ---- 6. Sample 20-component MDN ----
        (local.set $max_pi (f32.const -1e30))
        (local.set $k (i32.const 0))
        (loop $pi1
          (local.set $s (f32.mul (f32.load (i32.add (i32.const 33280) (i32.shl (local.get $k) (i32.const 2)))) (local.get $b1)))
          (f32.store (i32.add (i32.const 34048) (i32.shl (local.get $k) (i32.const 2))) (local.get $s)) ;; PIS = 0x8500 = 34048
          (if (f32.gt (local.get $s) (local.get $max_pi)) (then (local.set $max_pi (local.get $s))))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br_if $pi1 (i32.lt_u (local.get $k) (i32.const 20))))

        (local.set $sum_pi (f32.const 0.0))
        (local.set $k (i32.const 0))
        (loop $pi2
          (local.set $s (call $expf (f32.sub (f32.load (i32.add (i32.const 34048) (i32.shl (local.get $k) (i32.const 2)))) (local.get $max_pi))))
          (f32.store (i32.add (i32.const 34048) (i32.shl (local.get $k) (i32.const 2))) (local.get $s))
          (local.set $sum_pi (f32.add (local.get $sum_pi) (local.get $s)))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br_if $pi2 (i32.lt_u (local.get $k) (i32.const 20))))

        (local.set $sum2 (f32.const 0.0))
        (local.set $k (i32.const 0))
        (loop $pi3
          (local.set $s (f32.div (f32.load (i32.add (i32.const 34048) (i32.shl (local.get $k) (i32.const 2)))) (local.get $sum_pi)))
          (if (f32.lt (local.get $s) (f32.const 0.01)) (then (local.set $s (f32.const 0.0))))
          (f32.store (i32.add (i32.const 34048) (i32.shl (local.get $k) (i32.const 2))) (local.get $s))
          (local.set $sum2 (f32.add (local.get $sum2) (local.get $s)))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br_if $pi3 (i32.lt_u (local.get $k) (i32.const 20))))

        (local.set $r_pick (f32.mul (call $rng_u) (local.get $sum2)))
        (local.set $chosen (i32.const 19))
        (local.set $cum (f32.const 0.0))
        (local.set $k (i32.const 0))
        (block $pick_done
          (loop $pi4
            (local.set $cum (f32.add (local.get $cum) (f32.load (i32.add (i32.const 34048) (i32.shl (local.get $k) (i32.const 2))))))
            (if (f32.le (local.get $r_pick) (local.get $cum))
              (then (local.set $chosen (local.get $k)) (br $pick_done)))
            (local.set $k (i32.add (local.get $k) (i32.const 1)))
            (br_if $pi4 (i32.lt_u (local.get $k) (i32.const 20)))))

        (local.set $s1 (f32.max (call $expf (f32.sub (f32.load (i32.add (i32.const 33360) (i32.shl (local.get $chosen) (i32.const 2)))) (local.get $bias))) (f32.const 1e-4)))
        (local.set $s2 (f32.max (call $expf (f32.sub (f32.load (i32.add (i32.const 33440) (i32.shl (local.get $chosen) (i32.const 2)))) (local.get $bias))) (f32.const 1e-4)))
        (local.set $rho (f32.min (f32.max (call $tanhf (f32.load (i32.add (i32.const 33520) (i32.shl (local.get $chosen) (i32.const 2))))) (f32.const -0.9999)) (f32.const 0.9999)))
        (local.set $mu1 (f32.load (i32.add (i32.const 33600) (i32.shl (local.get $chosen) (i32.const 2)))))
        (local.set $mu2 (f32.load (i32.add (i32.const 33680) (i32.shl (local.get $chosen) (i32.const 2)))))
        (local.set $es  (call $sigmoid (f32.load (i32.const 33760))))
        (if (f32.lt (local.get $es) (f32.const 0.01)) (then (local.set $es (f32.const 0.0))))

        ;; Box-Muller z1, z2
        (local.set $u1 (f32.max (call $rng_u) (f32.const 1e-6)))
        (local.set $u2 (call $rng_u))
        (local.set $rad (f32.sqrt (f32.mul (f32.const -2.0) (call $logf (local.get $u1)))))
        (local.set $th  (f32.mul (f32.const 6.2831853) (local.get $u2)))
        (local.set $z1  (f32.mul (local.get $rad) (call $cosf (local.get $th))))
        (local.set $z2  (f32.mul (local.get $rad) (call $cosf (f32.sub (local.get $th) (f32.const 1.5707963)))))

        (local.set $dx  (f32.add (local.get $mu1) (f32.mul (local.get $s1) (local.get $z1))))
        (local.set $dy  (f32.add (local.get $mu2)
          (f32.mul (local.get $s2)
            (f32.add (f32.mul (local.get $rho) (local.get $z1))
                     (f32.mul (f32.sqrt (f32.add (f32.sub (f32.const 1.0) (f32.mul (local.get $rho) (local.get $rho))) (f32.const 1e-6))) (local.get $z2))))))
        (local.set $eos (select (f32.const 1.0) (f32.const 0.0) (f32.lt (call $rng_u) (local.get $es))))

        ;; Once the word body reached trailing space (S_AT_END=1) and this is a new pen-up jump (prev S_X2 > 0.5):
        ;; allow delayed back-strokes over the word (t/f crossbars, i/j dots) to finish,
        ;; and stop BEFORE emitting if the pen leaps forward into next-word space!
        (if (i32.and (global.get $S_AT_END) (f32.gt (global.get $S_X2) (f32.const 0.5)))
          (then
            (if (i32.or
                  (i32.ge_s (global.get $S_BACK_CNT) (i32.const 3))
                  (i32.or
                    (f32.gt (local.get $dx) (f32.const 3.5))
                    (f32.gt (f32.add (global.get $S_CUR_X) (local.get $dx))
                            (f32.add (global.get $S_MAX_X) (f32.const 1.5)))))
              (then (global.set $S_DONE (i32.const 1)) (br $done_blk)))
            (global.set $S_BACK_CNT (i32.add (global.get $S_BACK_CNT) (i32.const 1)))))

        (global.set $S_CUR_X (f32.add (global.get $S_CUR_X) (local.get $dx)))
        (if (i32.eqz (global.get $S_AT_END))
          (then
            (if (f32.gt (global.get $S_CUR_X) (global.get $S_MAX_X))
              (then (global.set $S_MAX_X (global.get $S_CUR_X))))))

        (global.set $S_X0 (local.get $dx))
        (global.set $S_X1 (local.get $dy))
        (global.set $S_X2 (local.get $eos))

        ;; Write [dx, dy, eos] to OUT_BUF[out_n * 3] (0xA000 = 40960)
        (local.set $op (i32.add (i32.const 40960) (i32.mul (local.get $out_n) (i32.const 12))))
        (f32.store (local.get $op) (local.get $dx))
        (f32.store offset=4 (local.get $op) (local.get $dy))
        (f32.store offset=8 (local.get $op) (local.get $eos))
        (local.set $out_n (i32.add (local.get $out_n) (i32.const 1)))
        (global.set $S_STEP (i32.add (global.get $S_STEP) (i32.const 1)))

        ;; Input format: tail_text + word + "   \0" -> first trailing space is at U - 4.
        ;; Arm S_AT_END on first pen lift at U - 4 (letting delayed crossbars/dots run); hard-stop at U - 2.
        (if (f32.gt (local.get $eos) (f32.const 0.5))
          (then
            (if (i32.ge_s (local.get $best_u) (i32.sub (local.get $U) (i32.const 2)))
              (then (global.set $S_DONE (i32.const 1)) (br $done_blk)))
            (if (i32.ge_s (local.get $best_u) (i32.sub (local.get $U) (i32.const 4)))
              (then (global.set $S_AT_END (i32.const 1))))))

        (br $step_loop)))

    (select (i32.sub (i32.const 0) (local.get $out_n)) (local.get $out_n) (global.get $S_DONE))
  )
)
