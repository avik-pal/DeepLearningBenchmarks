import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from benchmarking_gpu import run_benchmarks

n_iters = 101
batch_size = 8

print("The models are being run on input size of {}".format((batch_size, 3, 224, 224)))
print("The time is the average over {} iterations".format(n_iters-1))
run_benchmarks([resnet18, resnet34, resnet50, resnet101, resnet152], batch_size, n_iters)
