#include "ce.h"
#include <math.h>
#include <stddef.h>

#define BCE_T 18.0


// single label -> multiclass -> one hot
inline double ce_one_hot(const double* restrict z, size_t C, size_t y_idx) {
    if (!z || C==0 || y_idx>=C) return -1.0;

    // m = max zj
    double m = z[0];
    for (size_t j=1; j<C; j++) if (z[j]>m) m = z[j];

    /* CE = -log py = lse - zy
    LSE = m + lse(zj-m)
    p: softmax */

    // LSE
    double sum = 0.0;
    for (size_t j=0; j<C; j++) sum += exp(z[j]-m);
    double lse = m + log(sum);
    
    // CE
    return lse - z[y_idx];
}

// single label -> multiclass -> smoothed
inline double ce_smoothed(const double* restrict z, size_t C, size_t y_idx, const double eps){
    if (!z || C==0 || y_idx >= C) return -1.0;
    // LSE
    double m=z[0], s=0.0, sumz=0.0;
    for(size_t j=1;j<C;++j) if(z[j]>m) m=z[j];
    for(size_t j=0;j<C;++j){ s += exp(z[j]-m); sumz += z[j]; }
    double lse = m + log(s);    
    // CE = LSE - (1-eps)*z[y] - (eps/C)*sum(z)
    return lse - (1.0-eps)*z[y_idx] - (eps/(double)C)*sumz;
}

// single label -> binary
double bce(const double x, const double y) {
    double ax = fabs(x);
    // BCE = max(x,0) - xy + log(1+exp(-|x|))
    return (x>0 ? x : 0.0) - x*(double)y+log1p(exp(-ax));
}

inline double softplus_stable(const double x) {
    const double a = fabs(x);
    return (x > 0 ? x : 0.0) + log1p(exp(-a));
}

inline double bce_approx(const double x, const double y) {
    if (x > BCE_T)  return x - x*y;          // softplus(approx) = x
    if (x < -BCE_T)  return -x*y;            // softplus(approx) = 0
    return softplus_stable(x) - x*y;                
}

// multil label, each element each BCE, nomask
inline double bce_multilabel_nomask_sum(
    const double* restrict x,
    const double* restrict y,
    size_t K)
{
    double sum = 0.0;
    for (size_t k=0; k<K; ++k) {
        sum += bce_approx(x[k], y[k]);
    };
    return sum;
}

inline double bce_multilabel_nomask_mean(
    const double* restrict x,
    const double* restrict y,
    size_t K)
{
    if (K == 0) return 0.0;
    return bce_multilabel_nomask_sum(x, y, K) / (double)K;
}

// multil label, each element each BCE, mask
inline double bce_multilabel_mask_sum(
    const double* restrict x,       
    const double* restrict y,       
    const unsigned char* restrict m,
    size_t K)
{
    double sum = 0.0;
    for (size_t k=0; k<K; ++k) {
        if (!m[k]) continue;
        sum += bce_approx(x[k],y[k]);
    };
    return sum;
}

inline double bce_multilabel_mask_mean(
    const double* restrict x,
    const double* restrict y,
    const unsigned char* restrict m,
    size_t K)
{
    double sum = 0.0;
    size_t valid = 0;
    for (size_t k=0; k<K; ++k) {
        if (!m[k]) continue;
        sum += bce_approx(x[k],y[k]);
        ++valid;
    };
    return (valid > 0) ? (sum / (double)valid) : 0.0;
}