import torch, time

def run_benchmarks(arr, batch_size, n_iters):

    for m in arr:
        print("*"*10, "Model {}".format(m), "*"*10)
        forward_pass_time = []
        backward_pass_time = []
        total_time = []
        inference_time = []
        model = m()
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        grad_tensor = torch.ones(batch_size, 1000)

        for i in range(n_iters):

            start_time_forward = time.time()
            result = model(input_tensor)
            end_time_forward = time.time()

            start_time_backward = time.time()
            result.backward(grad_tensor)
            end_time_backward = time.time()

            if i is 1: # Ignore first iteration
                continue
            forward_pass_time.append(end_time_forward - start_time_forward)
            backward_pass_time.append(end_time_backward - start_time_backward)
            total_time.append(forward_pass_time[-1] + backward_pass_time[-1])

        print("Forward Pass Time : {}".format(min(forward_pass_time)))
        print("Backward Pass Time : {}".format(min(backward_pass_time)))
        print("Total Time : {}".format(min(total_time)))

        model.eval()
        for i in range(n_iters):
            start_time_inference = time.time()
            result = model(input_tensor)
            end_time_inference = time.time()

            if i is 1: # Ignore first iteration
                continue
            inference_time.append(end_time_forward - start_time_forward)

        print("Inference Time : {}".format(min(inference_time)))
