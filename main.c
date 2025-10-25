#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tensor.h"
#include "model.h"
#include "linear_layer.h"
#include "activations.h"
#include "losses.h"
#include "optimizer.h"

int main() {
    srand(time(NULL));
    double inputs_data[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double targets_data[4][1] = {{0}, {1}, {1}, {1}};

    Tensor* inputs[4];
    Tensor* targets[4];
    for (int i = 0; i < 4; ++i) {
        inputs[i] = create_tensor(2);
        targets[i] = create_tensor(1);
        inputs[i]->data[0] = inputs_data[i][0];
        inputs[i]->data[1] = inputs_data[i][1];
        targets[i]->data[0] = targets_data[i][0];
    }

    Model* model = create_model();
    model_add_layer(model, create_linear_layer(2, 3));
    model_add_layer(model, create_sigmoid_layer());
    model_add_layer(model, create_linear_layer(3, 1));
    model_add_layer(model, create_sigmoid_layer());

    Layer* loss_layer = create_mse_loss_layer();
    double learning_rate = 0.1;
    int epochs = 10000;

    printf("학습 시작...\n");

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0;
        for (int i = 0; i < 4; ++i) {
            Tensor* prediction = model_forward(model, inputs[i]);
            mse_loss_set_target(loss_layer, targets[i]);
            Tensor* loss = loss_layer->forward(loss_layer, prediction);
            total_loss += loss->data[0];
            zero_grad(model);
            Tensor* loss_grad = loss_layer->backward(loss_layer, NULL);
            model_backward(model, loss_grad);
            sgd_step(model, learning_rate);
            free_tensor(prediction);
            free_tensor(loss);
        }

        if ((epoch + 1) % 1000 == 0) {
            printf("Epoch %d/%d, Loss: %f\n", epoch + 1, epochs, total_loss / 4.0);
        }
    }

    printf("\n학습 완료. 결과 테스트:\n");

    for (int i = 0; i < 4; ++i) {
        Tensor* prediction = model_forward(model, inputs[i]);
        printf("입력: [%.1f, %.1f] -> 예측: %.5f (정답: %.1f)\n",
               inputs[i]->data[0], inputs[i]->data[1],
               prediction->data[0], targets[i]->data[0]);
        free_tensor(prediction);
    }

    for (int i = 0; i < 4; ++i) {
        free_tensor(inputs[i]);
        free_tensor(targets[i]);
    }
    free_model(model);
    if (loss_layer->params) free_tensor(loss_layer->params);
    free(loss_layer);

    return 0;
}