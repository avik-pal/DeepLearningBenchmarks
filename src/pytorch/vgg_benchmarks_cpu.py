import torch
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn
import time

batch_size = 8

print("The models are being run on input size of {}".format((batch_size, 3, 224, 224)))
print("The time is the average over {} iterations".format(n_iters-1))
for m in [vgg16, vgg16_bn, vgg19, vgg19_bn]:
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

    print("Forward Pass Time : {}".format(sum(forward_pass_time)/(n_iters-1)))
    print("Backward Pass Time : {}".format(sum(backward_pass_time)/(n_iters-1)))
    print("Total Time : {}".format(sum(total_time)/(n_iters-1)))

    model.eval()
    for i in range(n_iters):
        start_time_inference = time.time()
        result = model(input_tensor)
        end_time_inference = time.time()

        if i is 1: # Ignore first iteration
            continue
        inference_time.append(end_time_forward - start_time_forward)

    print("Inference Time : {}".format(sum(inference_time)/(n_iters-1)))
