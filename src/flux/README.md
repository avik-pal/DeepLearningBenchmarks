# Dependencies

1. Flux.jl
2. CuArrays.jl
3. BenchmarkTools.jl

NOTE:
1. For the benchmarks of `BatchNorm` models like `vgg16_bn` and `vgg19_bn` use the `cudnn_batchnorm` branch in https://github.com/avik-pal/Flux.jl till it is merged into the Flux master.
2. The `Conv` Algorithm benchmarks are somewhat biased. They use a `relu` in the forward pass however, in the backward pass the activation is treated as `identity`.
3. For reproducing the `Conv` Results checkout the `convbias` branch in https://github.com/avik-pal/CuArrays.jl.
