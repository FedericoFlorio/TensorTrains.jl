# Fourier basis functions
f_n(n::Int, P::Float64) = x -> exp(-im*2π/P*n*x)
F_n(n::Int, P::Float64) = x -> exp(im*2π/P*n*x)

function offset_fourier_freqs(tensors::Vector{Array{Complex{F},N}}, ax::Vector{Int64}) where {F<:Number, N}
    A = map(tensors) do Aᵗ
        for i in ax
            K = (size(Aᵗ)[i]-1)/2
            isinteger(K) ? K=Int(K) : throw(ArgumentError("Wrong dimension for axis of coefficients"))
            oldaxes = axes(Aᵗ)
            newaxes = [oldaxes[begin:i-1]..., -K:K, oldaxes[i+1:end]...]
            Aᵗ = OffsetArray(Aᵗ, newaxes...)
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
mutable struct FourierTensorTrain{F1<:Number, N} <: BasisTensorTrain{F1,N}
    tensors::Vector{OffsetArray{Complex{F1}, N, Array{Complex{F1}, N}}}
    z::Logarithmic{F2} where {F2<:Number}

    function FourierTensorTrain{F1,N}(tensors::Vector{OffsetArray{Complex{F1}, N, Array{Complex{F1}, N}}}; z::Logarithmic{F2}) where {F1<:Number, F2<:Number, N}
        any(any(isnan, a) for a in tensors) && error("NaN in Fourier Tensor Train")
        return new{F1,N}(tensors, z)
    end
end
function FourierTensorTrain(tensors::Vector{Array{Complex{F1},N}};
    z=Logarithmic(one(F1)), ax::Vector{Int64}=[3]) where {F1<:Number, N}
    N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
    return FourierTensorTrain{F1,N}(offset_fourier_freqs(tensors, ax); z)
end

@forward FourierTensorTrain.tensors Base.getindex, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.length, Base.eachindex,  
    check_bond_dims


function offset_fourier_freqs(A::FourierTensorTrain{F,N}) where {F<:Number, N}
    return FourierTensorTrain(collect.(A.tensors), z=A.z)
end
  
"""
    flat_fourier_tt(bondsizes::AbstractVector{<:Integer}, q...; imaginary=0.0)
    flat_fourier_tt(d::Integer, L::Integer, q...; imaginary=0.0)

Construct a (normalized) Fourier Tensor Train filled with a constant, by specifying either:
- `bondsizes`: the size of each bond, or
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
Additionally, an imaginary part can be provided through the parameter `imaginary`
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
    orthogonalize_right!(A::FourierTensorTrain; svd_trunc::SVDTrunc)

Brings `A` to right-orthogonal form by means of SVD decompositions.

Optionally performs truncations by passing a `SVDTrunc`.
"""
function orthogonalize_right!(A::FourierTensorTrain{F,N}; svd_trunc=TruncThresh(1e-6)) where {F,N}
    C = getproperty.(A.tensors, :parent) |> TensorTrain
    orthogonalize_right!(C; svd_trunc)
    B = FourierTensorTrain(C.tensors, z = A.z*C.z)
    A.tensors = B.tensors
    A.z = B.z
    return A
end

"""
    orthogonalize_left!(A::FourierTensorTrain; svd_trunc::SVDTrunc)

Brings `A` to left-orthogonal form by means of SVD decompositions.

Optionally performs truncations by passing a `SVDTrunc`.
"""
function orthogonalize_left!(A::FourierTensorTrain{F,N}; svd_trunc=TruncThresh(1e-6)) where {F,N}
    C = getproperty.(A.tensors, :parent) |> TensorTrain
    orthogonalize_left!(C; svd_trunc)
    B = FourierTensorTrain(C.tensors, z = A.z*C.z)
    A.tensors = B.tensors
    A.z = B.z
    return A
end

"""
    compress!(A::FourierTensorTrain{F,N}; svd_trunc::SVDTrunc)

Compresses `A` by means of SVD decompositions + truncations
"""
function compress!(A::FourierTensorTrain{F,N}; svd_trunc=TruncThresh(1e-6),
    is_orthogonal::Symbol=:none) where {F<:Number, N}
    if is_orthogonal == :none
        orthogonalize_right!(A, svd_trunc=TruncThresh(0.0))
        orthogonalize_left!(A, svd_trunc=svd_trunc)
    elseif is_orthogonal == :left
        orthogonalize_right!(A; svd_trunc)
    elseif is_orthogonal == :right
        orthogonalize_left!(A; svd_trunc)
    else
        throw(ArgumentError("Keyword `is_orthogonal` only supports: :none, :left, :right, got :$is_orthogonal"))
    end
    return A
end


# used to do stuff like `A+B` with `A,B` Fourier tensor trains
function _compose(f, A::FourierTensorTrain{F,NA}, B::FourierTensorTrain{F,NB}) where {F,NA,NB}
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
    FourierTensorTrain(tensors)
end


"""
    FourierTensorTrain_spin(A::TensorTrain{F,N}, K::Int, d::Int, P::Float64, σ::Float64) where {F,N}

Computes a Fourier tensor train starting from a Tensor Train A, in which each tensor has three axes. The third index ``x`` of each tensor (the physical one) is assumed to represent the values for a spin ``s``, with the convention ``x=1 ⟺ s=-1`` and ``x=2 ⟺ s=+1``.
For the purpose of going to a continuous domain, each matrix ``Aᵗ[m,n,:]`` is approximated as a linear combination of gaussians with variance ``σ²``, centered in 1 and -1: ``Aᵗ[m,n,1] g(-1,σ²) + Aᵗ[m,n,2] g(1,σ²)``.
The extra `d` parameter provides the possibility to scale the domain of the spin (i.e. define a spin with values ±1/d).
"""
function FourierTensorTrain_spin(A::TensorTrain{U,N}, K::Int, scale::Real, P::Real, σ::Real) where {U,N}
    N<3 && throw(ArgumentError("Tensors must have at least three axes"))
    any(!=(2), [size(Aᵗ)[3] for Aᵗ in A]) && throw(ArgumentError("Third axis of tensors for spins must have dimension 2"))

    k = OffsetVector([2π/P*α for α in -K:K], -K:K)
    expon = OffsetVector([exp(-k[α]^2*σ^2)/P for α in -K:K], -K:K)
    cos_kn = [expon[α] * cos(k[α]/scale) for α in -K:K]
    sin_kn = [expon[α] * sin(k[α]/scale) for α in -K:K]
    
    F = map(eachindex(A)) do t
        Aᵗ = reshape(A[t], size(A[t])[1:3]..., prod(size(A[t])[4:end]))
        @tullio Fᵗ[m,n,α,x] := (Aᵗ[m,n,1,x]+Aᵗ[m,n,2,x]) * cos_kn[α] + im * (Aᵗ[m,n,1,x]-Aᵗ[m,n,2,x]) * sin_kn[α]
        return reshape(Fᵗ, size(A[t])[1], size(A[t])[2], 2K+1, size(A[t])[4:end]...)
    end

    
    FTT = FourierTensorTrain(F, z=A.z)
    normalize_eachmatrix!(FTT)
    return FTT
end

"""
    evaluate(A::FourierTensorTrain{F,N}, X::Vector{<:Real}, P::Float64; normalize::Bool=true) where {F,N}

Evaluates `A` at input `X`, i.e. calculates ``∏ₜ Aᵗ(Xᵗ) = ∏ₜ ∑ₙ Ãᵗₙ Fₙ(Xᵗ)``, where ``Ãᵗₙ`` is the matrix of coefficients relative to the ``n``-th Fourier basis function.
"""
function evaluate(A::FourierTensorTrain{F,N}, X::Vector{<:Real}, P::Float64; normalize::Bool=true) where {F,N}
    L = Matrix(1.0I, size(A[begin],1), size(A[begin],1))
    z = Logarithmic(one(F))

    for (Aᵗ,xᵗ) in zip(A,X)
        K = (size(A[begin])[3]-1)/2 |> Int
        Fnx = OffsetVector([F_n(n,P)(xᵗ) for n in -K:K], -K:K)
        @tullio Bt[i,j] := Aᵗ[i,j,n] * Fnx[n]
        Bᵗ = real.(Bt)

        L = L * Bᵗ
        m = maximum(abs,L)
        if normalize && !iszero(m)
            L ./= m
            z *= m
        end
    end
    z *= tr(L)
    return z
end


function accumulate_L(A::FourierTensorTrain{F,N}, normalize::Bool=true) where {F<:Number, N}
    Lt = Matrix(1.0I, size(A[begin],1), size(A[begin],1))
    z = Logarithmic(1.0)

    L = map(At for At in A) do At
        # @tullio Lt[i,k] = Lt[i,j] * At[j,k,0]
        Lt = Lt * At[:,:,0]
        m = maximum(abs,Lt)
        if normalize && !iszero(m)
            Lt ./= m
            z *= m
        end
        return Lt
    end
    z *= tr(abs.(Lt))

    return L,z
end

function accumulate_R(A::FourierTensorTrain{F,N}, normalize::Bool=true) where {F<:Number, N}
    Rt = Matrix(1.0I, size(A[end],2), size(A[end],2))
    z = Logarithmic(1.0)

    R = map(At for At in Iterators.reverse(A)) do At
        # @tullio Rt[i,k] = At[i,j,0] * Rt[j,k]
        Rt = At[:,:,0] * Rt
        m = maximum(abs,Rt)
        if normalize && !iszero(m)
            Rt ./= m
            z *= m
        end
        return Rt
    end
    z *= tr(abs.(Rt))

    return R,z
end

"""
    marginals_Fourier(A::AbstractTensorTrain; l, r)

Computes the Fourier coefficients of the marginal distributions ``p(xᵗ)`` for each variable ``xᵗ``

### Optional arguments
- `l = accumulate_L(A)[1]`, `r = accumulate_R(A)[1]` pre-computed partial normalizations
"""
function marginals_Fourier(A::FourierTensorTrain{F,N};
    l = accumulate_L(A)[1], r = accumulate_R(A)[1], normalize::Bool=true) where {F<:Number,N}
    T = length(A)

    map(eachindex(A)) do t
        R = t+1 ≤ T ? r[T-t] : Matrix(I, size(A[end],2), size(A[end],2))
        L = t-1 ≥ 1 ? l[t-1] : Matrix(I, size(A[begin],1), size(A[begin],1))

        Aᵗ = A[t]
        @tullio lA[a¹,aᵗ⁺¹,x] := L[a¹,aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x]
        @tullio pFᵗ[x] := lA[a¹,aᵗ⁺¹,x] * R[aᵗ⁺¹,a¹]

        norm2 = sum(abs2,pFᵗ)
        pFᵗ ./= sqrt(norm2)
    end
end
#=
"""Why do we take r[t+1] and not r[T-t] in the original one?"""
"""
julia> function myf(A)
           Rt=""
           R = map(At for At in Iterators.reverse(A)) do A
                  Rt = A*Rt
           end
       end
myf (generic function with 1 method)

julia> A = ["a","b","c","d"]
4-element Vector{String}:
 "a"
 "b"
 "c"
 "d"

julia> myf(A)
4-element Vector{String}:
 "d"
 "cd"
 "bcd"
 "abcd"
 ;
 """
=#

function marginals(A::FourierTensorTrain{F,N}, P::Float64) where {F<:Number,N}
    K = (size(A[begin])[3]-1)/2 |> Int
    pF = marginals_Fourier(A)
    map(pF) do pFᵗ
        norm2 = sum(abs2,pFᵗ)/P
        pFᵗ ./= sqrt(norm2)
        x -> sum(pFᵗ[n]*F_n(n,P)(x) for n in -K:K) |> real
    end
end
function marginals(A::FourierTensorTrain{F,N}) where {F,N}
    error("The period of the basis function must be specified")
end