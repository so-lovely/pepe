// bench_bce_vs_approx.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

// 네 구현을 한 번에 보이게 하기 위해 소스 포함.
// 원하면 이 줄을  #include "ce.h"  로 바꾸고 ce.c를 별도로 컴파일/링크해도 됩니다.
#include "ce.c"

// ───────── 난수/유틸 ─────────
static inline uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double urand01(uint64_t *s) {
    return (splitmix64(s) >> 11) * (1.0/9007199254740992.0); // [0,1)
}
static inline double urand_range(uint64_t *s, double lo, double hi) {
    return lo + (hi - lo) * urand01(s);
}

// 단조 증가 타이머 (macOS 10.12+ / Linux OK)
static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000000000ull + (uint64_t)ts.tv_nsec;
}

// ───────── 설정 ─────────
typedef struct {
    size_t N;     // 샘플 수
    int iters;    // 반복 횟수(노이즈 평균화)
    int soft;     // 0: y∈{0,1}, 1: y∈[0,1]
} cfg_t;

static void gen_data(double *x, double *y, const cfg_t *cfg, uint64_t seed) {
    uint64_t s = seed ? seed : 0x12345678abcdefULL;
    for (size_t i=0;i<cfg->N;++i) {
        // 로짓을 넓게 뿌려 극단/중앙 모두 밟기 (근사 구간도 충분히 포함)
        x[i] = urand_range(&s, -500.0, 500.0);
        if (cfg->soft) {
            y[i] = urand01(&s);                 // 소프트 라벨 [0,1]
        } else {
            y[i] = (urand01(&s) < 0.5) ? 0.0 : 1.0; // 하드 라벨 {0,1}
        }
    }
}

// 하나의 함수에 대한 시간/체크섬 측정
static double run_bench(double (*fn)(double,double),
                        const double *x, const double *y,
                        const cfg_t *cfg, double *time_ms_out)
{
    volatile double sink = 0.0;  // 루프 제거(죽은 코드 최적화) 방지
    uint64_t t0 = now_ns();
    for (int it=0; it<cfg->iters; ++it) {
        for (size_t i=0;i<cfg->N;++i) {
            sink += fn(x[i], y[i]);
        }
    }
    uint64_t t1 = now_ns();
    if (time_ms_out) *time_ms_out = (t1 - t0)/1e6;
    return (double)sink;
}

int main(int argc, char** argv) {
    cfg_t cfg = {.N=3*1000*1000, .iters=1, .soft=0}; // 기본값: 3M, 하드 라벨
    // 사용법: ./bench [N] [iters] [--soft]
    if (argc > 1) cfg.N     = (size_t)strtoull(argv[1], NULL, 10);
    if (argc > 2) cfg.iters = atoi(argv[2]);
    if (argc > 3 && strcmp(argv[3],"--soft")==0) cfg.soft = 1;

    printf("Benchmark: bce(x,y) vs bce_approx(x,y)\n");
    printf("N=%zu, iters=%d, mode=%s\n",
           cfg.N, cfg.iters, cfg.soft ? "soft y in [0,1]" : "hard y in {0,1}");

    double *x = (double*)malloc(sizeof(double)*cfg.N);
    double *y = (double*)malloc(sizeof(double)*cfg.N);
    if (!x || !y) { fprintf(stderr, "alloc failed\n"); return 1; }

    gen_data(x, y, &cfg, 0);

    // (선택) 워밍업 — 캐시 예열/부동 노이즈 감소
    (void)run_bench(bce, x, y, &cfg, NULL);

    // 실제 벤치
    double t_bce=0.0, t_app=0.0;
    double sum_bce  = run_bench(bce,        x, y, &cfg, &t_bce);
    double sum_app  = run_bench(bce_approx, x, y, &cfg, &t_app);

    // 수치 일치성(최대 절대 차이) 확인: 한 번 더 순회
    double max_abs_diff = 0.0;
    for (size_t i=0;i<cfg.N;++i) {
        double v1 = bce(x[i], y[i]);
        double v2 = bce_approx(x[i], y[i]);
        double d = fabs(v1 - v2);
        if (d > max_abs_diff) max_abs_diff = d;
    }

    printf("[bce       ] time = %.3f ms, checksum = %.6f\n", t_bce, sum_bce);
    printf("[bce_approx] time = %.3f ms, checksum = %.6f\n", t_app, sum_app);
    printf("max |bce - bce_approx| = %.3e\n", max_abs_diff);

    free(x); free(y);
    return 0;
}
