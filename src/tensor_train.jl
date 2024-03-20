"""
    TensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}

A type for representing a Tensor Train
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
"""
struct TensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}
    tensors::Vector{Array{F,N}}

    function TensorTrain{F,N}(tensors::Vector{Array{F,N}}) where {F<:Number, N}
        N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        return new{F,N}(tensors)
    end
end
function TensorTrain(tensors::Vector{Array{F,N}}) where {F<:Number, N} 
    return TensorTrain{F,N}(tensors)
end


@forward TensorTrain.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    check_bond_dims, length, eachindex


  
"""
    flat_tt(bondsizes::AbstractVector{<:Integer}, q...)
    flat_tt(d::Integer, L::Integer, q...)

Construct a Tensor Train full of 1's, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function flat_tt(bondsizes::AbstractVector{<:Integer}, q...)
    TensorTrain([ones(bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])
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
function rand_tt(bondsizes::AbstractVector{<:Integer}, q...)
    TensorTrain([rand(bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])
end
rand_tt(d::Integer, L::Integer, q...) = rand_tt([1; fill(d, L-1); 1], q...)


"""
    orthogonalize_right!(A::AbstractTensorTrain; svd_trunc::SVDTrunc)

Bring `A` to right-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
"""
function orthogonalize_right!(C::TensorTrain; svd_trunc=TruncThresh(1e-6))
    Cᵀ = _reshape1(C[end])
    q = size(Cᵀ, 3)
    @cast M[m, (n, x)] := Cᵀ[m, n, x]
    D = fill(1.0,1,1,1)

    for t in length(C):-1:2
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := V'[m, (n, x)] x ∈ 1:q
        C[t] = _reshapeas(Aᵗ, C[t])     
        Cᵗ⁻¹ = _reshape1(C[t-1])
        @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
        @cast M[m, (n, x)] := D[m, n, x]
    end
    C[begin] = _reshapeas(D, C[begin])
    return C
end

"""
    orthogonalize_left!(A::AbstractTensorTrain; svd_trunc::SVDTrunc)

Bring `A` to left-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
"""
function orthogonalize_left!(C::TensorTrain; svd_trunc=TruncThresh(1e-6))
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[(m, x), n] |= C⁰[m, n, x]
    D = fill(1.0,1,1,1)

    for t in 1:length(C)-1
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := U[(m, x), n] x ∈ 1:q
        C[t] = _reshapeas(Aᵗ, C[t])
        Cᵗ⁺¹ = _reshape1(C[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Cᵗ⁺¹[l, n, x]
        @cast M[(m, x), n] |= D[m, n, x]
    end
    C[end] = _reshapeas(D, C[end])
    return C
end


# used to do stuff like `A+B` with `A,B` tensor trains
function _compose(f, A::TensorTrain{F,NA}, B::TensorTrain{F,NB}) where {F,NA,NB}
    @assert NA == NB
    @assert length(A) == length(B)
    tensors = map(zip(eachindex(A),A,B)) do (t,Aᵗ,Bᵗ)
        sa = size(Aᵗ); sb = size(Bᵗ)
        if t == 1
            Cᵗ = [ hcat(Aᵗ[:,:,x...], f(Bᵗ[:,:,x...])) 
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

