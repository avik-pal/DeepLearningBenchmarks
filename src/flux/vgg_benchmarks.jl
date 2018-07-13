include("vggmodels.jl")

using Flux, CuArrays, BenchmarkTools

batch_size = 8

for m in [vgg16, vgg16_bn, vgg19, vgg19_bn]
    println("*"^10,m,"*"^10)
    model = m() |> gpu
    input_arr = rand(224, 224, 3, batch_size) |> gpu
    grad_arr = ones(1000, batch_size) |> gpu

    @btime begin
        result = $model($input_arr)
        CUDAnative.synchronize()
    end

    result = model(input_arr)

    @btime begin
        Flux.back!($result, $grad_arr)
        CUDAnative.synchronize()
    end
end
