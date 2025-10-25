#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include <stddef.h>

typedef struct {
    Layer** layers;
    size_t num_layers;
    size_t capacity;
} Model;

Model* create_model();
void model_add_layer(Model* model, Layer* layer);
Tensor* model_forward(Model* model, Tensor* input);
void model_backward(Model* model, Tensor* loss_grad);
void free_model(Model* model);

#endif