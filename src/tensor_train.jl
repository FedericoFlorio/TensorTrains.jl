"""
    TensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}

A type for representing a Tensor Train
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
"""
mutable struct TensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}
    tensors::Vector{Array{F,N}}
    z::Logarithmic{F1} where {F1}

    function TensorTrain{F,N}(tensors::Vector{Array{F,N}}; z::Logarithmic{F1}=Logarithmic(abs(one(F)))) where {F<:Number, N, F1}
        N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        return new{F,N}(tensors, z)
    end
end
function TensorTrain(tensors::Vector{Array{F,N}}; z::Logarithmic{F1}=Logarithmic(abs(one(F)))) where 
{F<:Number, N, F1}
    return TensorTrain{F,N}(tensors; z)
end


@forward TensorTrain.tensors Base.getindex, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.length, Base.eachindex,  
    check_bond_dims

  
"""
    flat_tt(bondsizes::AbstractVector{<:Integer}, q...)
    flat_tt(d::Integer, L::Integer, q...)

Construct a (normalized) Tensor Train filled with a constant, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function flat_tt(bondsizes::AbstractVector{<:Integer}, q...)
    L = length(bondsizes) - 1
    x = 1 / (prod(bondsizes)^(1/L)*prod(q))
    TensorTrain([fill(x, bondsizes[t], bondsizes[t+1], q...) for t in 1:L])
end
flat_tt(d::Integer, L::Integer, q...) = flat_tt([1; fill(d, L-1); 1], q...)

"""
    rand_tt(bondsizes::AbstractVector{<:Integer}, q...)
    rand_tt(d::Integer, L::Integer, q...)

Construct a Tensor Train with entries random in [0,1], by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function rand_tt(bondsizes::AbstractVector{<:Integer}, q...; rng=default_rng())
    TensorTrain([rand(rng, bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])
end
rand_tt(d::Integer, L::Integer, q...; kw...) = rand_tt([1; fill(d, L-1); 1], q...; kw...)


"""
    orthogonalize_right!(A::AbstractTensorTrain; svd_trunc::SVDTrunc)

Bring `A` to right-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
"""
function orthogonalize_right!(C::TensorTrain{F}; svd_trunc=TruncThresh(1e-6)) where F
    length(C)==1 && return C
    Cᵀ = _reshape1(C[end])
    @cast M[m, (n, x)] := Cᵀ[m, n, x]
    D = fill(1.0,1,1,1)
    c = Logarithmic(abs(one(F)))

    for t in length(C):-1:2
        U, λ, V = svd_trunc(M)
        q = prod(size(C[t])[3:end])
        @cast Aᵗ[m, n, x] := V'[m, (n, x)] x ∈ 1:q
        s = (size(Aᵗ,1), size(Aᵗ,2), size(C[t])[3:end]...)
        C[t] = reshape(Aᵗ, s...)
        Cᵗ⁻¹ = _reshape1(C[t-1])
        @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
        m = maximum(abs, D)
        if !isnan(m) && !isinf(m) && !iszero(m)
            D ./= m
            c *= m
        end
        @cast M[m, (n, x)] := D[m, n, x]
    end
    s = (size(D,1), size(D,2), size(C[begin])[3:end]...)
    C[begin] = reshape(D, s...)
    C.z /= c
    return C
end

"""
    orthogonalize_left!(A::AbstractTensorTrain; svd_trunc::SVDTrunc)

Bring `A` to left-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
"""
function orthogonalize_left!(C::TensorTrain{F}; svd_trunc=TruncThresh(1e-6)) where F
    length(C)==1 && return C
    C⁰ = _reshape1(C[begin])
    @cast M[(m, x), n] := C⁰[m, n, x]
    D = fill(1.0,1,1,1)
    c = Logarithmic(abs(one(F)))

    for t in 1:length(C)-1
        U, λ, V = svd_trunc(M)
        q = prod(size(C[t])[3:end])
        @cast Aᵗ[m, n, x] := U[(m, x), n] x ∈ 1:q
        s = (size(Aᵗ,1), size(Aᵗ,2), size(C[t])[3:end]...)
        C[t] = reshape(Aᵗ, s...)
        Cᵗ⁺¹ = _reshape1(C[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Cᵗ⁺¹[l, n, x]
        m = maximum(abs, D)
        if !isnan(m) && !isinf(m) && !iszero(m)
            D ./= m
            c *= m
        end
        @cast M[(m, x), n] := D[m, n, x]
    end
    s = (size(D,1), size(D,2), size(C[end])[3:end]...)
    C[end] = reshape(D, s...)
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

