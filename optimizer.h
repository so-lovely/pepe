#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "model.h"

void sgd_step(Model* model, double learning_rate);
void zero_grad(Model* model);

#endif