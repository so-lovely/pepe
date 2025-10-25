#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor* create_tensor(size_t size) {
    if (size == 0) return NULL;

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->size = size;
    t->data = (double*)calloc(size, sizeof(double));
    t->grad = (double*)calloc(size, sizeof(double));

    if (!t->data || !t->grad) {
        free(t->data);
        free(t->grad);
        free(t);
        return NULL;
    }
    return t;
}

void free_tensor(Tensor* t) {
    if (t != NULL) {
        free(t->data);
        free(t->grad);
        free(t);
    }
}
