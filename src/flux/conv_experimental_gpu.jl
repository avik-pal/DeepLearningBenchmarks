using CuArrays, Flux
using CuArrays.CUDNN: @check, libcudnn, cudnnStatus_t, libcudnn_handle,
  cudnnDataType, TensorDesc, FilterDesc
using Flux: relu
using Flux.Tracker: TrackedArray
using CUDAnative
using CuArrays: @cuindex, cudims
import Flux.Tracker
import Flux.Tracker: data, istracked, track, unbroadcast, @grad, nobacksies
using NNlib: padtuple, cdims, dilation_dims
using CuArrays.CUDNN: cudnnConvolutionBackwardBias, cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
  cudnnActivationBackward, cudnnConvolutionBiasActivationForward

CuParam{T,N} = Union{CuArray{T,N},TrackedArray{T,N,CuArray{T,N}}}

function convbias!(y::CuArray{T}, x::CuArray{T}, w::CuArray{T}, b::CuArray{T};
                   pad = 0, stride = 1, mode = 0,
                   alpha = 1, dilation = 1, activationMode = 5) where T<:Union{Float32,Float64}
  all(x -> x == 1, dilation) || error("Only dilation = 1 is supported in CuArrays")
  cudnnConvolutionBiasActivationForward(y, x, w, b, padding=pad, stride=stride, mode=mode, alpha1=alpha, activationMode=activationMode)
end

function convbias(x::CuArray{T}, w::CuArray{T}, b::CuArray{T};
                  pad = 0, stride = 1, mode = 0,
                  alpha = 1, dilation = 1) where T<:Union{Float32,Float64}
  pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
  convbias!(similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_)),
            x, w, b, pad = pad_, stride = stride_, dilation = dilation, activationMode = 1) # We are using relu as identity requires CUDNN 7.1+
end

∇conv_bias(Δ::CuArray{T}, b::CuArray{T}; pad = 0, beta = 0,
           stride = 1, mode = 0, alpha = 1, dilation = 1) where T<:Union{Float32,Float64} =
  reshape(cudnnConvolutionBackwardBias(similar(b), Δ, alpha=alpha, beta=beta), :)

function (m::Flux.Conv)(x::Union{CuParam{T,4},CuParam{T,5}})  where T<:Union{Float32,Float64}
  result = convbias(x, m.weight, m.bias, pad = m.pad, stride = m.stride, dilation = m.dilation)
  m.σ == identity ? result : m.σ.(result) # Replace with cudnn function calls
end

convbias(x::TrackedArray, w::TrackedArray, b::TrackedArray; kw...) = track(convbias, x, w, b; kw...)

convbias(x::CuArray{T}, w::TrackedArray, b::TrackedArray; kw...) where T<:Union{Float32,Float64} =
  track(convbias, x, w, b; kw...)

convbias(x::CuArray{T}, w::CuArray{T}, b::TrackedArray; kw...) where T<:Union{Float32,Float64} =
  track(convbias, x, w, b; kw...)

convbias(x::TrackedArray, w::CuArray{T}, b::TrackedArray; kw...) where T<:Union{Float32,Float64} =
  track(convbias, x, w, b; kw...)

convbias(x::TrackedArray, w::TrackedArray, b::CuArray{T}; kw...) where T<:Union{Float32,Float64} =
  track(convbias, x, w, b; kw...)

convbias(x::CuArray{T}, w::TrackedArray, b::CuArray{T}; kw...) where T<:Union{Float32,Float64} =
  track(convbias, x, w, b; kw...)

convbias(x::TrackedArray, w::CuArray{T}, b::CuArray{T}; kw...) where T<:Union{Float32,Float64} =
  track(convbias, x, w, b; kw...)

@grad function convbias(x, w, b; kw...)
  bias = reshape(b, map(_->1, kw[2][2])..., :, 1)
  y = convbias(data.((x, w, bias))...; kw...)
  y, Δ -> (istracked(x) ? NNlib.∇conv_data(data.((Δ, x, w))...; kw...) : nothing,
           istracked(w) ? NNlib.∇conv_filter(data.((Δ, x, w))...; kw...) : nothing,
           istracked(b) ? ∇conv_bias(data.((Δ, bias))...; kw...) : nothing)
end
