module TensorTrains

using MKL
using Lazy: @forward
using TensorCast: @cast, TensorCast
using LinearAlgebra: svd, norm, tr, I, dot, normalize!
using LinearAlgebra
using OffsetArrays
using LogarithmicNumbers: Logarithmic
using Tullio: @tullio
using Random: AbstractRNG, GLOBAL_RNG, default_rng
using StatsBase: sample!, sample
using StatsBase

export 
    getindex, iterate, firstindex, lastindex, setindex!, eachindex, length, show,
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    AbstractTensorTrain, TensorTrain, normalize_eachmatrix!, +, -, ==, isapprox, evaluate, 
    bond_dims, flat_tt, rand_tt, orthogonalize_right!, orthogonalize_left!, compress!,
    marginals, twovar_marginals, lognormalization, normalization, normalize!, 
    dot, norm, norm2m,
    sample!, sample,
    PeriodicTensorTrain, flat_periodic_tt, rand_periodic_tt,
    BasisTensorTrain, FourierTensorTrain, flat_fourier_tt, rand_fourier_tt, orthogonalize_right, orthogonalize_left, compress, FourierTensorTrain_spin, marginals_Fourier

include("utils.jl")
include("svd_trunc.jl")
include("abstract_tensor_train.jl")
include("tensor_train.jl")
include("periodic_tensor_train.jl")

include("basis_tensor_trains/basis_tensor_train.jl")
include("basis_tensor_trains/fourier_tensor_train.jl")


end # end module
