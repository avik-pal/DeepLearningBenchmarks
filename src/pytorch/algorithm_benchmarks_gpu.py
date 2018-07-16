import torch, time
import torch.nn as nn

n_iters = 101

def run_benchmarks(input_tensor, grad_tensor, layer):
    global n_iters
    forward_pass_time = []
    backward_pass_time = []
    total_time = []

    for i in range(n_iters):
        start_time_forward = time.time()
        result = layer(input_tensor)
        torch.cuda.synchronize()
        end_time_forward = time.time()

        start_time_backward = time.time()
        result.backward(grad_tensor)
        torch.cuda.synchronize()
        end_time_backward = time.time()

        if i is 1: # Ignore first iteration
            continue
        forward_pass_time.append(end_time_forward - start_time_forward)
        backward_pass_time.append(end_time_backward - start_time_backward)
        total_time.append(forward_pass_time[-1] + backward_pass_time[-1])

    print("Forward Pass Time : {}".format(sum(forward_pass_time)/(n_iters-1)))
    print("Backward Pass Time : {}".format(sum(backward_pass_time)/(n_iters-1)))
    print("Total Time : {}".format(sum(total_time)/(n_iters-1)))

input_tensor = torch.rand(1, 3, 224, 224).cuda()
grad_tensor = torch.ones(1, 64, 224, 224).cuda()
layer = nn.Conv2d(3, 64, 3, padding = 1).cuda()
print("Benchmarks for Conv3x3/1")
run_benchmarks(input_tensor, grad_tensor, layer)

layer = nn.Conv2d(3, 64, 5, padding = 2).cuda()
print("Benchmarks for Conv5x5/1")
run_benchmarks(input_tensor, grad_tensor, layer)

layer = nn.Conv2d(3, 64, 3, padding = 1, stride = 2).cuda()
grad_tensor = torch.ones(1, 64, 112, 112).cuda()
print("Benchmarks for Conv3x3/2")
run_benchmarks(input_tensor, grad_tensor, layer)

layer = nn.Conv2d(3, 64, 5, padding = 2, stride = 2).cuda()
print("Benchmarks for Conv5x5/2")
run_benchmarks(input_tensor, grad_tensor, layer)

layer = nn.MaxPool2d(3, stride = 2, padding = 1).cuda()
grad_tensor = torch.ones(1, 3, 112, 112).cuda()
print("Benchmarks for Maxpool")
run_benchmarks(input_tensor, grad_tensor, layer)

layer = nn.AvgPool2d(3, stride = 2, padding = 1).cuda()
print("Benchmarks for Meanpool")
run_benchmarks(input_tensor, grad_tensor, layer)

grad_tensor = torch.ones(1, 3, 224, 224).cuda()
layer = nn.BatchNorm2d(3).cuda()
print("Benchmarks for BatchNorm")
run_benchmarks(input_tensor, grad_tensor, layer)

input_tensor = torch.rand(1, 1024).cuda()
grad_tensor = torch.ones(1, 512).cuda()
layer = nn.Linear(1024, 512).cuda()
print("Benchmarks for Dense")
run_benchmarks(input_tensor, grad_tensor, layer)
