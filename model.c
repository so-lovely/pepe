#include "model.h"
#include <stdlib.h>

Model* create_model() {
    Model* model = (Model*)malloc(sizeof(Model));
    model->num_layers = 0;
    model->capacity = 4;
    model->layers = (Layer**)malloc(model->capacity * sizeof(Layer*));
    return model;
}

void model_add_layer(Model* model, Layer* layer) {
    if (model->num_layers >= model->capacity) {
        model->capacity *= 2;
        model->layers = (Layer**)realloc(model->layers, model->capacity * sizeof(Layer*));
    }
    model->layers[model->num_layers++] = layer;
}

Tensor* model_forward(Model* model, Tensor* input) {
    Tensor* current_output = input;
    for (size_t i = 0; i < model->num_layers; ++i) {
        current_output = model->layers[i]->forward(model->layers[i], current_output);
    }
    return current_output;
}

void model_backward(Model* model, Tensor* loss_grad) {
    Tensor* current_grad = loss_grad;
    for (int i = model->num_layers - 1; i >= 0; --i) {
        Tensor* prev_grad = model->layers[i]->backward(model->layers[i], current_grad);
        if (current_grad != loss_grad) {
            free_tensor(current_grad);
        }
        current_grad = prev_grad;
    }
    free_tensor(current_grad);
}

void free_model(Model* model) {
    if (model) {
        for (size_t i = 0; i < model->num_layers; ++i) {
            Layer* l = model->layers[i];
            if (l->params) free_tensor(l->params);
            if (l->internal_state && l->free_internal_state) {
                l->free_internal_state(l);
            }
            free(l);
        }
        free(model->layers);
        free(model);
    }
}