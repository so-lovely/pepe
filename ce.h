#ifndef CE_H
#define CE_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

double ce_one_hot(const double* logits, size_t C, size_t y);
double ce_smoothed(const double* z, size_t C, size_t y_idx, const double eps);

double bce(double x, double y);
double bce_approx(const double x, const double y);

double bce_multilabel_nomask_sum(
    const double* restrict x,
    const double* restrict y,
    size_t K);
double bce_multilabel_nomask_mean(
    const double* restrict x,
    const double* restrict y,
    size_t K);
double bce_multilabel_mask_sum(
    const double* restrict x,       
    const double* restrict y,       
    const unsigned char* restrict m,
    size_t K);
double bce_multilabel_mask_mean(
    const double* restrict x,
    const double* restrict y,
    const unsigned char* restrict m,
    size_t K);
    
#ifdef __cplusplus
}
#endif
#endif