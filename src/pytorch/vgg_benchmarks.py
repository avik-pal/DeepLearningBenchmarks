import torch
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn
from benchmarking_gpu import run_benchmarks
import time

n_iters = 101
batch_size = 8

print("The models are being run on input size of {}".format((batch_size, 3, 224, 224)))
print("The time is the average over {} iterations".format(n_iters-1))
run_benchmarks([vgg16, vgg16_bn, vgg19, vgg19_bn])
