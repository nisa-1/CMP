mutable struct HOTRG_y{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "Central tensor"
    T::TT

    function HOTRG_y(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(T)
    end
end


function _step_hotrg_y(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        Ux::AbstractTensorMap{E, S, 2, 1}
    ) where {E, S}

    @tensor T[-1 -2; -3 -4] :=
        conj(Ux[1 2; -1]) * Ux[3 4; -4] * A2[1 5; -3 3] * A1[2 -2; 5 4]
    return T
end
#=
function _step_hotrg_x(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        Uy::AbstractTensorMap{E, S, 2, 1}
    ) where {E, S}

    @tensor T[-1 -2; -3 -4] :=
        A1[-1 1; 3 5] * A2[5 2; 4 -4] * conj(Uy[1 2; -2]) * Uy[3 4; -3]
    return T
end

function _get_hotrg_xproj(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        trunc::MatrixAlgebraKit.TruncationStrategy
    ) where {E, S}

    @plansor MM[-1 -2; -3 -4] :=
        A2[-1 5; 1 2] * A1[-2 3; 5 4] *
        conj(A2[-3 6; 1 2]) * conj(A1[-4 3; 6 4])
    # Project to hermitian because gross_neveu tests would not pass without
    _, U, ε = eigh_trunc!(project_hermitian!(MM); trunc = trunc)

    # get right unitary
    @plansor MM[-1 -2; -3 -4] :=
        conj(A2[2 5; 1 -1]) * conj(A1[4 3; 5 -2]) *
        A2[2 6; 1 -3] * A1[4 3; 6 -4]
    # Project to hermitian because gross_neveu tests would not pass without
    _, U′, ε′ = eigh_trunc!(project_hermitian!(MM); trunc = trunc)

    if ε > ε′
        U, ε = U′, ε′
    end
    return U, ε
end
=#
function _get_hotrg_yproj(
        A1::TensorMap{E, S, 2, 2}, A2::TensorMap{E, S, 2, 2},
        trunc::MatrixAlgebraKit.TruncationStrategy
    ) where {E, S}

    @plansor MM[-1 -2; -3 -4] :=
        A1[1 -1; 2 5] * A2[5 -2; 4 3] *
        conj(A1[1 -3; 2 6]) * conj(A2[6 -4; 4 3])
    # Project to hermitian because gross_neveu tests would not pass without
    _, U, ε = eigh_trunc!(project_hermitian!(MM); trunc = trunc)

    # get top unitary
    @plansor MM[-1 -2; -3 -4] :=
        conj(A1[1 2; -1 5]) * conj(A2[5 4; -2 3]) *
        A1[1 2; -3 6] * A2[6 4; -4 3]
    # Project to hermitian because gross_neveu tests would not pass without
    _, U′, ε′ = eigh_trunc!(project_hermitian!(MM); trunc = trunc)

    if ε > ε′
        U, ε = U′, ε′
    end
    return U, ε
end

function step!(scheme::HOTRG_y, trunc::MatrixAlgebraKit.TruncationStrategy)
    Ux, = _get_hotrg_xproj(scheme.T, scheme.T, trunc)
    scheme.T = _step_hotrg_y(scheme.T, scheme.T, Ux)
    Uy, = _get_hotrg_yproj(scheme.T, scheme.T, trunc)
    scheme.T = _step_hotrg_x(scheme.T, scheme.T, Uy)
    return scheme
end

function Base.show(io::IO, scheme::HOTRG_y)
    println(io, "HOTRG - Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
