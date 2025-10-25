#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    double* data;
    double* grad;
    size_t size;
} Tensor;

Tensor* create_tensor(size_t size);
void free_tensor(Tensor* t);

#endif