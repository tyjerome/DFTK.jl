# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/temperature)
# DOS (density of states)
# D(ε) = N'(ε)
#
# LDOS (local density of states)
# LD(ε) = sum_n f_n' |ψn|^2 = sum_n δ(ε - ε_n) |ψn|^2

"""
Total density of states at energy ε
"""
function compute_dos(ε, basis, eigenvalues; smearing=basis.model.smearing,
                     temperature=basis.model.temperature)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_dos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    D = zeros(typeof(ε), basis.model.n_spin_components)
    for σ = 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
            D[σ] -= (filled_occ * basis.kweights[ik] / temperature
                     * Smearing.occupation_derivative(smearing, enred))
        end
    end
    D = mpi_sum(D, basis.comm_kpts)
end
function compute_dos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
        compute_dos(ε, scfres.basis, scfres.eigenvalues; kwargs...)
end

"""
Local density of states, in real space. `weight_threshold` is a threshold
to screen away small contributions to the LDOS.
"""
function compute_ldos(ε, basis::PlaneWaveBasis{T}, eigenvalues, ψ;
                      smearing=basis.model.smearing,
                      temperature=basis.model.temperature,
                      weight_threshold=eps(T)) where {T}
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_ldos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    weights = deepcopy(eigenvalues)
    for (ik, εk) in enumerate(eigenvalues)
        for (iband, εnk) in enumerate(εk)
            enred = (εnk - ε) / temperature
            weights[ik][iband] = (-filled_occ / temperature
                                  * Smearing.occupation_derivative(smearing, enred))
        end
    end

    # Use compute_density routine to compute LDOS, using just the modified
    # weights (as "occupations") at each k-point. Note, that this automatically puts in the
    # required symmetrization with respect to kpoints and BZ symmetry
    compute_density(basis, ψ, weights; occupation_threshold=weight_threshold)
end
function compute_ldos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
    compute_ldos(ε, scfres.basis, scfres.eigenvalues, scfres.ψ; kwargs...)
end

"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_ldos end

"""
Projected density of states at energy ε
"""
# PD(ε) = sum_n f_n' |<ψn,ϕ>|^2
function compute_pdos(ε, basis, eigenvalues, projs;
    smearing=basis.model.smearing,
    temperature=basis.model.temperature)

    if (temperature == 0) || smearing isa Smearing.None
        error("compute_dos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    D = zeros(typeof(ε), basis.model.n_spin_components)
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
            D[σ] -= (filled_occ * basis.kweights[ik] * projs[ik][iband] / temperature
                     *
                     Smearing.occupation_derivative(smearing, enred))
        end
    end
    D = mpi_sum(D, basis.comm_kpts)
end

function compute_pdos(ε, basis, eigenvalues, ψ, i::Integer, l::Integer, psp, position; kwargs...)
    projs = compute_pdos_projs(basis, eigenvalues, ψ, i, l, psp, position)

    pdos = map(x->zeros(typeof(ε[1]),length(ε), 2l + 1), 1:basis.model.n_spin_components)
    for j = 1 : 2l+1
        projs_lj = [projsk[:,j] for projsk in projs]
        for (i,εi) in enumerate(ε)
            pdos_ij = compute_pdos(εi, basis, eigenvalues, projs_lj; kwargs...)
            for n_spin = 1 : basis.model.n_spin_components
                pdos[n_spin][i,j] = pdos_ij[n_spin]
            end
        end
    end

   pdos
end

function compute_pdos(ε, i::Integer, l::Integer,
    iatom::Integer, basis, eigenvalues, ψ; kwargs...)
    compute_pdos(ε, basis, eigenvalues, ψ, i, l, basis.model.atoms[iatom].psp, basis.model.positions[iatom]; kwargs...)
end

function compute_pdos(i::Integer, l::Integer, iatom::Integer, scfres::NamedTuple;
    ε=scfres.εF, kwargs...)
    scfres = unfold_bz(scfres)
    compute_pdos(ε, i, l, iatom, scfres.basis, scfres.eigenvalues, scfres.ψ; kwargs...)
end


function compute_pdos_projs(basis, eigenvalues, ψ, i::Integer, l::Integer, psp, position)

    projs = Vector{Matrix}(undef, length(eigenvalues))
    position = vector_red_to_cart(basis.model, position)

    G_plus_k_all = [to_cpu(Gplusk_vectors_cart(basis, basis.kpoints[ik])) for ik = 1:length(basis.kpoints)]
    fourier_form = atom_fourier_form(i,l,psp,G_plus_k_all)

    for (ik, ψk) in enumerate(ψ)
        fourier_form_k = fourier_form[ik]
        atom_shift = [dot(position, Gi) for Gi in G_plus_k_all[ik]]
        atom_fourier_k = exp.(-im * atom_shift) .* fourier_form_k ./ sqrt(basis.model.unit_cell_volume)
        ak_normalize = similar(atom_fourier_k)
        for m = 1:2l+1
            ak_normalize[:, m] = atom_fourier_k[:, m] ./ norm(atom_fourier_k[:, m])
        end
        projs[ik] = abs2.(ψk' * ak_normalize)
    end

    projs
end


"""
Build form factors (Fourier transforms of pseudo-atomic wavefunctions) for an atom centered at 0.
"""
function atom_fourier_form(i::Integer, l::Integer, psp, G_plus_k_all::Vector{Vector{Vec3{TT}}}) where {TT}
    T = real(TT)
    @assert psp isa PspUpf
    # Pre-compute the radial parts of the  pseudo-atomic wavefunctions at unique |p| to speed up
    # the form factor calculation (by a lot). Using a hash map gives O(1) lookup.
    
    radials = IdDict{T,T}()  # IdDict for Dual compatibility
    for (ik, G_plus_k) in enumerate(G_plus_k_all)
        for p in G_plus_k
            p_norm = norm(p)
            if !haskey(radials, p_norm)
                radials[p_norm] = eval_psp_pswfc_fourier(psp, i, l, p_norm)
            end
        end
    end

    form_factors = Vector{Matrix{Complex{T}}}(undef, length(G_plus_k_all))
    for (ik, G_plus_k) in enumerate(G_plus_k_all)
        form_factors_ik = Matrix{Complex{T}}(undef, length(G_plus_k), 2l + 1)
        for (ip, p) in enumerate(G_plus_k)
            radials_p = radials[norm(p)]
            count = 1
            for m = -l:l
                # see "Fourier transforms of centered functions" in the docs for the formula
                angular = (-im)^l * ylm_real(l, m, p)
                form_factors_ik[ip, m+l+1] = radials_p * angular
            end
        end
        form_factors[ik] = form_factors_ik
    end
    form_factors
end

function plot_pdos end
