# Fourier basis functions
f_n(n::Int, P::Float64) = x -> exp(-im*2π/P*n*x)
F_n(n::Int, P::Float64) = x -> exp(im*2π/P*n*x)

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

    function FourierTensorTrain{F,N}(tensors::Vector{OffsetArray{Complex{F}, N, Array{Complex{F}, N}}},
        z::Logarithmic{F}) where {F<:Number, N}
        return new{F,N}(tensors, z)
    end
end
function FourierTensorTrain(tensors::Vector{Array{Complex{F},N}};
    z=Logarithmic(one(F))) where {F<:Number, N}
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
# function orthogonalize_right(A::FourierTensorTrain{F,N}; svd_trunc=TruncThresh(1e-6)) where {F,N}
#     C = collect.(A.tensors)
#     Cᵀ = _reshape1(C[end])
#     q = size(Cᵀ, 3)
#     @cast M[m, (n, x)] := Cᵀ[m, n, x]
#     D = fill(1.0,1,1,1)
#     c = Logarithmic(one(F))

#     for t in length(C):-1:2
#         U, λ, V = svd_trunc(M)
#         @cast Aᵗ[m, n, x] := V'[m, (n, x)] x ∈ 1:q
#         C[t] = _reshapeas(Aᵗ, C[t])
#         Cᵗ⁻¹ = _reshape1(C[t-1])
#         @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
#         m = maximum(abs, D)
#         if !isnan(m) && !isinf(m) && !iszero(m)
#             D ./= m
#             c *= m
#         end
#         @cast M[m, (n, x)] := D[m, n, x]
#     end
#     C[begin] = _reshapeas(D, C[begin])
#     A.z /= c
#     return FourierTensorTrain(C, z=A.z)
# end
function orthogonalize_right!(A::FourierTensorTrain{F,N}; svd_trunc=TruncThresh(1e-6)) where {F,N}
    C = getproperty.(A.tensors, :parent) |> TensorTrain
    orthogonalize_right!(C)
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
    # for i in eachindex(A)
    #     println("Axes of A[$i] = $(axes(A[i]))")
    #     println("Axes of C[$i] = $(axes(C[i]))")
    # end
    orthogonalize_left!(C)
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
        orthogonalize_right!(A; svd_trunc=TruncThresh(0.0))
        orthogonalize_left!(A; svd_trunc)
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
The extra `scale` parameter provides the possibility to scale the domain of the spin (i.e. define a spin with values ±scale).
"""
function FourierTensorTrain_spin(A::TensorTrain{U,N}, K::Int, d::Int, P::Float64, σ::Float64) where {U,N}
    N!=3 && throw(ArgumentError("Tensors for spins must have three axes"))
    any(!=(2), [size(Aᵗ)[3] for Aᵗ in A]) && throw(ArgumentError("Third axis of tensors for spins must have dimension 2"))

    F = [Array{Complex{U}}(undef,size(Aᵗ)[1], size(Aᵗ)[2], 2K+1) for Aᵗ in A]

    k = OffsetVector([2π/P*n for n in -K:K], -K:K)
    expon = OffsetVector([exp(-k[n]^2*σ^2)/P for n in -K:K], -K:K)
    cos_kn = [expon[n] * cos(k[n]/d) for n in -K:K]
    sin_kn = [expon[n] * sin(k[n]/d) for n in -K:K]
    
    for t in eachindex(A)
        Aᵗ, Fᵗ = A[t], F[t]
        for m in axes(Aᵗ)[1], n in axes(Aᵗ)[2]
            Fᵗ[m,n,:] = cos_kn .+ im.*(Aᵗ[m,n,1]-Aᵗ[m,n,2]).*sin_kn
        end
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


# function evaluate_Fourier(C::Vector{Complex{F}}, x::U, P::Float64; normalized::Bool=true) where {F<:Number, U<:Real}
#     Fx = sum([C[n]*F_n(n,P)(x) for n in eachindex(F)])
#     if normalized
#         Fx /= (sum(abs2, C) / P)
#     end
#     return Fx |> real
# end

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