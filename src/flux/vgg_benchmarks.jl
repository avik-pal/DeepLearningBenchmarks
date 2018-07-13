include("vggmodels.jl")
include("benchmarking_gpu.jl")

using Flux, CuArrays, BenchmarkTools

batch_size = 8

run_benchmarks([vgg16, vgg16_bn, vgg19, vgg19_bn])
