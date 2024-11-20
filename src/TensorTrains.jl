module TensorTrains

using KrylovKit: eigsolve
using Lazy: @forward
using TensorCast: @cast, TensorCast
using LinearAlgebra: svd, norm, tr, I, dot, normalize!
using LinearAlgebra
using OffsetArrays
using LogarithmicNumbers: Logarithmic
using MKL
using MPSKit: InfiniteMPS, DenseMPO, VUMPS, approximate, dot, add_util_leg, site_type, physicalspace
using Random: AbstractRNG, default_rng
using StatsBase: StatsBase, sample!, sample
using TensorCast: @cast, TensorCast
using TensorKit: TensorMap, ⊗, ℝ, id, storagetype
using Tullio: @tullio
using OffsetArrays

export 
    getindex, iterate, firstindex, lastindex, setindex!, eachindex, length, show,
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    AbstractTensorTrain, BasisTensorTrain, TensorTrain, FourierTensorTrain, normalize_eachmatrix!, +, -, ==, isapprox, evaluate, 
    bond_dims, flat_tt, flat_fourier_tt, rand_tt, rand_fourier_tt, orthogonalize_right!, orthogonalize_left!, compress!,
    marginals, twovar_marginals, lognormalization, normalization, normalize!, 
    dot, norm, norm2m,
    sample!, sample,
    AbstractPeriodicTensorTrain, PeriodicTensorTrain, flat_periodic_tt, rand_periodic_tt,

    # Uniform Tensor Trains
    AbstractUniformTensorTrain, UniformTensorTrain, InfiniteUniformTensorTrain,
    symmetrized_uniform_tensor_train, periodic_tensor_train,
    flat_infinite_uniform_tt, rand_infinite_uniform_tt,
    TruncVUMPS


include("utils.jl")
include("svd_trunc.jl")
include("abstract_tensor_train.jl")
include("tensor_train.jl")
include("periodic_tensor_train.jl")

include("basis_tensor_trains/basis_tensor_train.jl")
include("basis_tensor_trains/fourier_tensor_train.jl")


end # end module
