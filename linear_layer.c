#include "linear_layer.h"
#include <stdlib.h>
#include <math.h>

typedef struct {
    Tensor* input;
} LinearState;

static Tensor* linear_forward(Layer* self, Tensor* input) {
    size_t output_size = self->params->size / (input->size + 1);
    size_t input_size = input->size;
    Tensor* output = create_tensor(output_size);
    double* weights = self->params->data;
    double* bias = self->params->data + (input_size * output_size);

    for (size_t j = 0; j < output_size; ++j) {
        output->data[j] = bias[j];
        for (size_t i = 0; i < input_size; ++i) {
            output->data[j] += input->data[i] * weights[j * input_size + i];
        }
    }

    LinearState* state = (LinearState*)malloc(sizeof(LinearState));
    state->input = input;
    self->internal_state = state;

    return output;
}

// dL/dx, dL/dW, dL/db
static Tensor* linear_backward(Layer* self, Tensor* output_grad) {
    LinearState* state = (LinearState*)self->internal_state;
    Tensor* input = state->input;
    
    size_t output_size = output_grad->size;
    size_t input_size = input->size;

    double* weights = self->params->data;
    double* weights_grad = self->params->grad;
    double* bias_grad = self->params->grad + (input_size * output_size);

    Tensor* input_grad = create_tensor(input_size);

    // dL/dW_ji = dL/dy_j * x_i
    for (size_t j = 0; j < output_size; ++j) {
        for (size_t i = 0; i < input_size; ++i) {
            weights_grad[j * input_size + i] += output_grad->data[j] * input->data[i];
        }
    }

    // dL/db_j = dL/dy_j
    for (size_t j = 0; j < output_size; ++j) {
        bias_grad[j] += output_grad->data[j];
    }

    // dL/dx_i = sum_j(dL/dy_j * W_ji)
    for (size_t i = 0; i < input_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            input_grad->data[i] += output_grad->data[j] * weights[j * input_size + i];
        }
    }
    
    return input_grad;
}

static void free_linear_state(Layer* self) {
    if (self && self->internal_state) {
        free(self->internal_state);
        self->internal_state = NULL;
    }
}

Layer* create_linear_layer(size_t input_size, size_t output_size) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->forward = linear_forward;
    layer->backward = linear_backward;
    layer->free_internal_state = free_linear_state;
    size_t params_size = input_size * output_size + output_size;
    layer->params = create_tensor(params_size);

    for (size_t i = 0; i < params_size; ++i) {
        layer->params->data[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    
    layer->internal_state = NULL;
    return layer;
}