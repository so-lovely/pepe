CC = gcc
CFLAGS = -Wall -Wextra -g -std=c99
LDFLAGS = -lm

SRCS = \
    tensor.c \
    layer.c \
    linear_layer.c \
    activations.c \
    losses.c \
    model.c \
    optimizer.c \
    main.c
OBJS = $(SRCS:.c=.o)

TARGET = simple_nn

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean