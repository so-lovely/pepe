#include "activations.h"
#include <stdlib.h>
#include <math.h>

typedef struct {
    Tensor* output;
} ActivationState;

static Tensor* sigmoid_forward(Layer* self, Tensor* input) {
    Tensor* output = create_tensor(input->size);
    for (size_t i = 0; i < input->size; ++i) {
        output->data[i] = 1.0 / (1.0 + exp(-input->data[i]));
    }

    ActivationState* state = (ActivationState*)malloc(sizeof(ActivationState));
    state->output = output;
    self->internal_state = state;

    return output;
}

static Tensor* sigmoid_backward(Layer* self, Tensor* output_grad) {
    ActivationState* state = (ActivationState*)self->internal_state;
    Tensor* output = state->output;
    Tensor* input_grad = create_tensor(output->size);

    for (size_t i = 0; i < output->size; ++i) {
        double sigmoid_deriv = output->data[i] * (1.0 - output->data[i]);
        input_grad->data[i] = output_grad->data[i] * sigmoid_deriv;
    }
    return input_grad;
}

static void free_activation_state(Layer* self) {
    if (self && self->internal_state) {
        ActivationState* state = (ActivationState*)self->internal_state;
        free_tensor(state->output);
        free(state);
        self->internal_state = NULL;
    }
}

Layer* create_sigmoid_layer() {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->forward = sigmoid_forward;
    layer->backward = sigmoid_backward;
    layer->free_internal_state = free_activation_state;
    layer->params = NULL;
    layer->internal_state = NULL;
    return layer;
}