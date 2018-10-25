# Popular Computer Vision Model Benchmarks

## Input Dimensions
1. Batch Size = 8, Image = 3 x 224 x 224 (IF NOTHING SPECIFIED / CPU USED)
2. Batch Size = 4, Image = 3 x 224 x 224
    * Resnet 101
    * Resnet 152

## GPU USED --- Titan 1080Ti 12 GB
|Model|Framework|Forward Pass|Backward Pass|Total Time|Inference|
|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG16|Pytorch 0.4.1|0.0245 s|0.0606 s|0.0852 s|0.0234 s|
||Flux|-|-|-|-|
|VGG16 BN|Pytorch 0.4.1|0.0271 s|0.0672 s|0.0943 s|0.0273 s|
||Flux|-|-|-|-|
|VGG19|Pytorch 0.4.1|0.0281 s|0.0741 s|0.1021 s|0.0280 s|
||Flux|-|-|-|-|
|VGG19 BN|Pytorch 0.4.1|0.0321 s|0.0812 s|0.1134 s|0.0325 s|
||Flux|-|-|-|-|
|Resnet18|Pytorch 0.4.1|0.0064 s|0.0125 s|0.0190 s|0.0050 s|
||Flux|-|-|-|-|
|Resnet34|Pytorch 0.4.1|0.0092 s|0.0216 s|0.0307 s|0.0092 s|
||Flux|-|-|-|-|
|Resnet50|Pytorch 0.4.1|0.0155 s|0.0351 s|0.0506 s|0.0152 s|
||Flux|-|-|-|-|
|Resnet101|Pytorch 0.4.1|0.0297 s|0.0379 s|0.0676 s|0.0298 s|
||Flux|-|-|-|-|
|Resnet152|Pytorch 0.4.1|0.0431 s|0.05337 s|0.0965 s|0.0429 s|
||Flux|-|-|-|-|

## CPU USED --- Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz
|Model|Framework|Forward Pass|Backward Pass|Total Time|Inference|
|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG16|Pytorch 0.4.1|__6.6024 s__|__9.4336 s__|__16.036 s__|__6.4216 s__|
||Flux|10.458 s|10.245 s|20.703 s|10.111 s|
|VGG16 BN|Pytorch 0.4.1|__7.0793 s__|__9.0536 s__|__16.132 s__|__6.7909 s__|
||Flux|29.633 s|18.649 s|49.282 s|24.047 s|
|VGG19|Pytorch 0.4.1|__8.3075 s__|__10.899 s__|__19.207 s__|__8.0593 s__|
||Flux|12.226 s|12.457 s|24.683 s|12.029 s|
|VGG19 BN|Pytorch 0.4.1|__8.7794 s__|__12.739 s__|__21.519 s__|__8.4044 s__|
||Flux|28.518 s|21.464 s|49.982 s|22.649 s|
<!-- |Resnet18|Pytorch 0.4.1|||||
||Flux|||||
|Resnet34|Pytorch 0.4.1|||||
||Flux|||||
|Resnet50|Pytorch 0.4.1|||||
||Flux|||||
|Resnet101|Pytorch 0.4.1|||||
||Flux|||||
|Resnet152|Pytorch 0.4.1|||||
||Flux||||| -->

# Individual Layer Benchmarks

## Layer Descriptions
1. Conv3x3/1 = Conv2d, 3x3 Kernel, 1x1 Padding, 1x1 Stride
2. Conv5x5/1 = Conv2d, 5x5 Kernel, 2x2 Padding, 1x1 Stride
3. Conv3x3/2 = Conv2d, 3x3 Kernel, 1x1 Padding, 2x2 Stride
4. Conv5x5/2 = Conv2d, 5x5 Kernel, 2x2 Padding, 2x2 Stride
5. Dense = 1024 => 512
6. BatchNorm = BatchNorm2d

## GPU USED --- Titan 1080Ti 12 GB
|Layer|Framework|Forward Pass|Backward Pass|Total Time|
|:---:|:---:|:---:|:---:|:---:|
|Conv3x3/1|Pytorch 0.4.1|0.2312 ms|__0.5359 ms__|__0.7736 ms__|
||Flux|__0.1984 ms__|0.7640 ms|0.9624 ms|
|Conv5x5/1|Pytorch 0.4.1|0.2667 ms|__0.5345 ms__|__0.8299 ms__|
||Flux|__0.2065 ms__|0.8075 ms|1.014 ms|
|Conv3x3/2|Pytorch 0.4.1|0.1170 ms|__0.2203 ms__|__0.3376 ms__|
||Flux|__0.0927 ms__|0.5988 ms|0.6915 ms|
|Conv5x5/2|Pytorch 0.4.1|0.1233 ms|__0.2162 ms__|__0.3407 ms__|
||Flux|__0.0941 ms__|0.6515 ms|0.7456 ms|
|Dense|Pytorch 0.4.1|0.0887 ms|__0.1523 ms__|__0.2411 ms__|
||Flux|__0.0432 ms__|0.2044 ms|0.2476 ms|
|BatchNorm|Pytorch 0.4.1|0.1096 ms|0.1999 ms|0.3095 ms|
||Flux|-|-|-|

<!-- ## CPU USED --- Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz
|Layer|Framework|Forward Pass|Backward Pass|Total Time|
|:---:|:---:|:---:|:---:|:---:|
|Conv3x3/1|Pytorch 0.4.1||||
||Flux||||
|Conv5x5/1|Pytorch 0.4.1||||
||Flux||||
|Conv3x3/2|Pytorch 0.4.1||||
||Flux||||
|Conv5x5/2|Pytorch 0.4.1||||
||Flux||||
|Dense|Pytorch 0.4.1||||
||Flux||||
|BatchNorm|Pytorch 0.4.1||||
||Flux|||| -->

# NOTE

To reproduce the benchmarks checkout `Flux` __master__ and `CuArrays` __master__.
Since the Batchnorm GPU is broken for Flux master so we cannot perform the benchmarks using that.
