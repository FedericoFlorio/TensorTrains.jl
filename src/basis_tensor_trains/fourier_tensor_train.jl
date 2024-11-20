# Fourier basis functions
f_n(n::Int, T::Float64) = x -> exp(-im*2π/T*n*x)
F_n(n::Int, T::Float64) = x -> exp(im*2π/T*n*x)

function offset_fourier_freqs(tensors::Vector{Array{Complex{F},N}}) where {F<:Number, N}
    A = map(tensors) do Aᵗ
        for i in 3:N
            K = (size(Aᵗ)[i]-1)/2
            isinteger(K) ? K=Int(K) : throw(ArgumentError("Wrong dimension for axis of coefficients"))
            oldsize = axes(Aᵗ)
            newsize = [oldsize[begin:i-1]..., -K:K, oldsize[i+1:end]...]
            Aᵗ = OffsetArray(Aᵗ, newsize...)
        end
        return Aᵗ
    end     
end
"""
    FourierTensorTrain{F<:Number, N} <: BasisTensorTrain{F,N}

A type for representing the approximation of a Matrix-product state in Fourier basis
- `F` is the type of the matrix entries (which are complex)
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
"""
mutable struct FourierTensorTrain{F<:Number, N} <: BasisTensorTrain{F,N}
    tensors::Vector{OffsetArray{Complex{F}, N, Array{Complex{F}, N}}}
    z::Logarithmic{F}

    function FourierTensorTrain{F,N}(tensors::Vector{OffsetArray{Complex{F}, N, Array{Complex{F}, N}}}, z::Logarithmic{F}) where {F<:Number, N}
        return new{F,N}(tensors, z)
    end
end
function FourierTensorTrain(tensors::Vector{Array{Complex{F},N}}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Number, N}
    N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
    return FourierTensorTrain{F,N}(offset_fourier_freqs(tensors), z)
end


@forward FourierTensorTrain.tensors Base.getindex, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.length, Base.eachindex,  
    check_bond_dims

  
"""
    flat_fourier_tt(bondsizes::AbstractVector{<:Integer}, q...; imaginary=0.0)
    flat_fourier_tt(d::Integer, L::Integer, q...; imaginary=0.0)

Construct a (normalized) Fourier Tensor Train filled with a constant, by specifying either:
- `bondsizes`: the size of each bond, or
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
Additionally, you can provide an imaginary part with the parameter `imaginary`
"""
function flat_fourier_tt(bondsizes::AbstractVector{<:Integer}, q...; imaginary=0.0)
    L = length(bondsizes) - 1
    x = 1 / (prod(bondsizes)^(1/L)*prod(q)) + imaginary*1im
    FourierTensorTrain([fill(x, bondsizes[t], bondsizes[t+1], q...) for t in 1:L])
end
flat_fourier_tt(d::Integer, L::Integer, q...; imaginary=0.0) = flat_fourier_tt([1; fill(d, L-1); 1], q..., imaginary=imaginary)

"""
    rand_fourier_tt(bondsizes::AbstractVector{<:Integer}, q...; imaginary=0.0)
    rand_fourier_tt(d::Integer, L::Integer, q...; imaginary=0.0)

Construct a Tensor Train with entries random in [0+0im,1+0im], by specifying either:
- `bondsizes`: the size of each bond, or
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function rand_fourier_tt(bondsizes::AbstractVector{<:Integer}, q...)
    FourierTensorTrain([rand(ComplexF64, bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])
end
rand_fourier_tt(d::Integer, L::Integer, q...) = rand_tt([1; fill(d, L-1); 1], q...)


"""
    orthogonalize_right!(A::AbstractTensorTrain; svd_trunc::SVDTrunc)

Bring `A` to right-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
"""
function orthogonalize_right!(C::FourierTensorTrain{F,N}; svd_trunc=TruncThresh(1e-6)) where {F,N}
    Cᵀ = _reshape1(C[end])
    q = size(Cᵀ, 3)
    @cast M[m, (n, x)] := Cᵀ[m, n, x]
    D = fill(1.0,1,1,1)
    c = Logarithmic(one(F))

    for t in length(C):-1:2
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := V'[m, (n, x)] x ∈ 1:q
        C[t] = _reshapeas(Aᵗ, C[t])     
        Cᵗ⁻¹ = _reshape1(C[t-1])
        @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
        m = maximum(abs, D)
        if !isnan(m) && !isinf(m) && !iszero(m)
            D ./= m
            c *= m
        end
        @cast M[m, (n, x)] := D[m, n, x]
    end
    C[begin] = _reshapeas(D, C[begin])
    C.z /= c
    return C
end

"""
    orthogonalize_left!(A::AbstractTensorTrain; svd_trunc::SVDTrunc)

Bring `A` to left-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
"""
function orthogonalize_left!(C::FourierTensorTrain{F,N}; svd_trunc=TruncThresh(1e-6)) where {F,N}
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[(m, x), n] |= C⁰[m, n, x]
    D = fill(1.0,1,1,1)
    c = Logarithmic(one(F))

    for t in 1:length(C)-1
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := U[(m, x), n] x ∈ 1:q
        C[t] = _reshapeas(Aᵗ, C[t])
        Cᵗ⁺¹ = _reshape1(C[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Cᵗ⁺¹[l, n, x]
        m = maximum(abs, D)
        if !isnan(m) && !isinf(m) && !iszero(m)
            D ./= m
            c *= m
        end
        @cast M[(m, x), n] |= D[m, n, x]
    end
    C[end] = _reshapeas(D, C[end])
    C.z /= c
    return C
end


# used to do stuff like `A+B` with `A,B` tensor trains
function _compose(f, A::TensorTrain{F,NA}, B::TensorTrain{F,NB}) where {F,NA,NB}
    @assert NA == NB
    @assert length(A) == length(B)
    tensors = map(zip(eachindex(A),A,B)) do (t,Aᵗ,Bᵗ)
        sa = size(Aᵗ); sb = size(Bᵗ)
        if t == 1
            Cᵗ = [ hcat(float(A.z) * Aᵗ[:,:,x...], float(B.z) * f(Bᵗ[:,:,x...])) 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), 1, sa[2]+sb[2], size(Aᵗ)[3:end]...)
        elseif t == lastindex(A)
            Cᵗ = [ vcat(Aᵗ[:,:,x...], Bᵗ[:,:,x...]) 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), sa[1]+sb[1], 1, size(Aᵗ)[3:end]...)
        else
            Cᵗ = [ [Aᵗ[:,:,x...] zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) Bᵗ[:,:,x...]] 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), (sa .+ sb)[1:2]..., size(Aᵗ)[3:end]...)
        end
    end
    TensorTrain(tensors)
end

