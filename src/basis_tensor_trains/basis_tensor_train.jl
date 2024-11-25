"""
    BasisTensorTrain

An abstract type representing the decomposition of a matrix-product state of continuous variables in a basis of functions.
As an example, consider the matrix-product state ``A(x¹,…,xᵀ) = ∏ₜ Aᵗ(xᵗ)``, with ``x¹,…,xᵀ`` continuous variables. Then, taken a family of orthonormal functions ``\\{uₐ(x)\\}ₐ``, the Basis tensor train approximating ``A(x¹,…,xᵀ)`` can be defined as ``B(x¹,…,xᵀ) = ∏ₜ Bᵗ(xᵗ) with Bᵗ(xᵗ) = ∑ₐ B̃ᵗₐ uₐ(xᵗ)``. In this case, the matrices ``B̃ᵗₐ`` contain the coefficient of the expansion of ``Bᵗ(xᵗ)`` in the basis ``\\{uₐ(x)\\}ₐ``.
It currently supports the Fourier basis ``uₐ(x) = exp(2π/T a x)``.
"""
abstract type BasisTensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N} end

Base.eltype(::BasisTensorTrain{F,N}) where {N,F} = F