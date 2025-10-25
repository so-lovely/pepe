#include "optimizer.h"
#include <string.h>

void sgd_step(Model* model, double learning_rate) {
    for (size_t i = 0; i < model->num_layers; ++i) {
        Tensor* params = model->layers[i]->params;
        if (params) {
            for (size_t j = 0; j < params->size; ++j) {
                params->data[j] -= learning_rate * params->grad[j];
            }
        }
    }
}

void zero_grad(Model* model) {
    for (size_t i = 0; i < model->num_layers; ++i) {
        Tensor* params = model->layers[i]->params;
        if (params) {
            memset(params->grad, 0, params->size * sizeof(double));
        }
    }
}