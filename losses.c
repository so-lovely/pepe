#include "losses.h"
#include <stdlib.h>

typedef struct {
    Tensor* prediction;
    Tensor* target;
} MSEState;

// L = 0.5 * sum((target - prediction)^2) / N
static Tensor* mse_forward(Layer* self, Tensor* prediction) {
    MSEState* state = (MSEState*)self->internal_state;
    Tensor* target = state->target;

    double total_loss = 0.0;
    for (size_t i = 0; i < prediction->size; ++i) {
        double error = target->data[i] - prediction->data[i];
        total_loss += 0.5 * error * error;
    }

    Tensor* loss_tensor = create_tensor(1);
    loss_tensor->data[0] = total_loss / prediction->size;

    state->prediction = prediction;
    return loss_tensor;
}

// dL/d(prediction) = (prediction - target) / N
static Tensor* mse_backward(Layer* self, Tensor* output_grad) {
    (void)output_grad;
    MSEState* state = (MSEState*)self->internal_state;
    Tensor* prediction = state->prediction;
    Tensor* target = state->target;
    Tensor* input_grad = create_tensor(prediction->size);

    for (size_t i = 0; i < prediction->size; ++i) {
        input_grad->data[i] = (prediction->data[i] - target->data[i]) / prediction->size;
    }
    return input_grad;
}

static void free_mse_state(Layer* self) {
    if (self && self->internal_state) {
        free(self->internal_state);
        self->internal_state = NULL;
    }
}

Layer* create_mse_loss_layer() {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->forward = mse_forward;
    layer->backward = mse_backward;
    layer->free_internal_state = free_mse_state;
    layer->params = NULL;
    layer->internal_state = NULL;
    return layer;
}

void mse_loss_set_target(Layer* loss_layer, Tensor* target) {
    if (loss_layer->internal_state) {
        loss_layer->free_internal_state(loss_layer);
    }
    MSEState* state = (MSEState*)malloc(sizeof(MSEState));
    state->target = target;
    state->prediction = NULL;
    loss_layer->internal_state = state;
}