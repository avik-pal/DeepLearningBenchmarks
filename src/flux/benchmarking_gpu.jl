function run_benchmarks(arr)
    for m in arr
        println("*"^10,m,"*"^10)
        model = m() |> gpu
        input_arr = rand(224, 224, 3, batch_size) |> gpu
        grad_arr = ones(1000, batch_size) |> gpu

        println("Forward Pass Time :")
        @btime begin
            result = $model($input_arr)
            CUDAnative.synchronize()
        end

        result = model(input_arr)

        println("Backward Pass Time :")
        @btime begin
            Flux.back!($result, $grad_arr)
            CUDAnative.synchronize()
        end
    end

    for m in arr
        println("*"^10,m,"*"^10)
        model = m()
        Flux.testmode!(model)
        input_arr = rand(224, 224, 3, batch_size) |> gpu
        grad_arr = ones(1000, batch_size) |> gpu

        println("Inference Time")
        @btime begin
            result = $model($input_arr)
            CUDAnative.synchronize()
        end
    end
end