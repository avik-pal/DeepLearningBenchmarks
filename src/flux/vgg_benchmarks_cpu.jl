include("vggmodels.jl")

using Flux, BenchmarkTools

batch_size = 8

for m in [vgg16, vgg16_bn, vgg19, vgg19_bn]
    println("*"^10,m,"*"^10)
    model = m() |> gpu
    input_arr = rand(224, 224, 3, batch_size)
    grad_arr = ones(1000, batch_size)

    @btime result = $model($input_arr

    result = model(input_arr)

    @btime Flux.back!($result, $grad_arr)
end
