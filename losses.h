#ifndef LOSSES_H
#define LOSSES_H

#include "layer.h"

Layer* create_mse_loss_layer();
void mse_loss_set_target(Layer* loss_layer, Tensor* target);

#endif