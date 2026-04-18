# ─── Makefile for Lab 1: Matrix Multiplication on CPU ───────────────────────
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -funroll-loops  -fopenmp-simd
LDFLAGS = -lm #-lpthread

# Targets
all: matmul

matmul: matmul.c
	$(CC) $(CFLAGS) matmul.c -o matmul $(LDFLAGS)

clean:
	rm -f matmul *.o

benchmark: matmul
	@echo "Running comprehensive benchmark..."
	./matmul

.PHONY: all clean run_matmul run_ml benchmark
