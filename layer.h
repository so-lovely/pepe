#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

typedef struct Layer Layer;

struct Layer {
    Tensor* (*forward)(Layer* self, Tensor* input);
    Tensor* (*backward)(Layer* self, Tensor* output_grad);
    Tensor* params;
    void* internal_state;
    void (*free_internal_state)(Layer* self);
};

#endif