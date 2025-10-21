#!/bin/sh
gcc -o run.out run3.c -lm -O3 -fopenmp -mavx512f -mavx512bw -mavx512vl -mavxvnni
