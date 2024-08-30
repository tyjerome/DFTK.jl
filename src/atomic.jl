# Build projection matrix for a single atom of index atom_index
# Stored as projs[k][lmi] = f_lmi,
# where K is running over all k-points, l, m are AM quantum numbers, 
# and i is running over all pseudo-atomic wavefunctions for a given l
# Note, that is built based on PspUpf pseudo-atomic wavefunctions,
# dose not orthogonalize all wavefunctions,
# requires symmetrization with respect to kpoints and BZ symmetry (now achieved by unfolding all the quantities)
# Maybe refactored in the future

# When nspin = 2, then n_orbitals = 2*(2l+1)
# Do orthogonalization

function atomic_wavefunction_trial(basis, atom_index)
    pos_cart = vector_red_to_cart(basis.model, basis.model.positions[atom_index])
    psp = basis.model.atoms[atom_index].psp
    @assert psp isa PspUpf

    G_plus_k_all = [to_cpu(Gplusk_vectors_cart(basis, basis.kpoints[ik])) for ik = 1:length(basis.kpoints)]
    # Build Fourier transform factors centered at 0.
    lmax = psp.lmax
    n_funs_per_l = [length(psp.r2_pswfcs[l+1]) for l in 0:lmax]
    eval_psp_fourier(i, l, p) = eval_psp_pswfc_fourier(psp, i, l, p)
    fourier_form = DFTK.nonlocal.atomic_centered_function_form_factors(eval_psp_fourier, G_plus_k_all, lmax, n_funs_per_l)
    map(1:length(basis.kpoints)) do ik
        fourier_form_ik = fourier_form[ik]
        structure_factor_ik = exp.(-im .*[dot(pos_cart, Gik) for Gik in G_plus_k_all[ik]])
        wfn = [structure_factor_ik .* fourier_form_ik[:, iproj] ./ sqrt(basis.model.unit_cell_volume) for iproj in 1:size(fourier_form_ik, 2)]
        for iproj in 1:size(wfn)
            wfn[iproj] = wfn[iproj]/ norm(wfn[iproj])  # Normalise but not orthogonalise (Why ?)
        end
        wfn_matrix = hcat(wfn)
        ortho_lowdin(wfn_matrix)
    end
end

#just put the same function here in order to avoid error
function atomic_centered_function_form_factors(fun::Function,
    G_plus_ks::AbstractVector{<:AbstractVector{Vec3{TT}}},
    lmax, n_funs_per_l) where {TT}
T = real(TT)

# Pre-compute the radial parts of the non-local atomic functions at unique |p| to speed up
# the form factor calculation (by a lot). Using a hash map gives O(1) lookup.

# Maximum number of atomic functions over angular momenta so that form factors
# for a given `p` can be stored in an `nfuns x (lmax + 1)` matrix.
n_funs_max = maximum(n_funs_per_l)

radials = IdDict{T,Matrix{T}}()  # IdDict for Dual compatibility
for G_plus_k in G_plus_ks
for p in G_plus_k
p_norm = norm(p)
if !haskey(radials, p_norm)
radials_p = Matrix{T}(undef, n_funs_max, lmax + 1)
for l = 0:lmax, ifuns_l = 1:n_funs_per_l[l+1]
radials_p[ifuns_l, l+1] = fun( ifuns_l, l, p_norm)
end
radials[p_norm] = radials_p
end
end
end

form_factors = Vector{Matrix{Complex{T}}}(undef, length(G_plus_ks))
n_funs = sum(l -> n_funs_per_l[l+1] * (2l + 1), 0:lmax; init=0)::Int
for (ik, G_plus_k) in enumerate(G_plus_ks)
form_factors_ik = Matrix{Complex{T}}(undef, length(G_plus_k), n_funs)
for (ip, p) in enumerate(G_plus_k)
radials_p = radials[norm(p)]
count = 1
for l = 0:lmax, m = -l:l
# see "Fourier transforms of centered functions" in the docs for the formula
angular = (-im)^l * ylm_real(l, m, p)
for ifuns_l = 1:n_funs_per_l[l+1]
form_factors_ik[ip, count] = radials_p[ifuns_l, l+1] * angular
count += 1
end
end
@assert count == n_funs + 1
end
form_factors[ik] = form_factors_ik
end

form_factors
end

#always keep a original version
function atomic_wavefunction(basis, atom_index)
    pos_cart = vector_red_to_cart(basis.model, basis.model.positions[atom_index])
    psp = basis.model.atoms[atom_index].psp
    @assert psp isa PspUpf

    G_plus_k_all = [to_cpu(Gplusk_vectors_cart(basis, basis.kpoints[ik])) for ik = 1:length(basis.kpoints)]
    # Build Fourier transform factors centered at 0.
    lmax = psp.lmax
    n_funs_per_l = [length(psp.r2_pswfcs[l+1]) for l in 0:lmax]
    eval_psp_fourier(i, l, p) = eval_psp_pswfc_fourier(psp, i, l, p) #eval_psp_projector_fourier
    fourier_form = atomic_centered_function_form_factors(eval_psp_fourier, G_plus_k_all, lmax, n_funs_per_l)
    map(1:length(basis.kpoints)) do ik
        fourier_form_ik = fourier_form[ik]
        structure_factor_ik = exp.(-im .*[dot(pos_cart, Gik) for Gik in G_plus_k_all[ik]])
        map(1:size(fourier_form_ik, 2)) do iproj
            wfn = structure_factor_ik .* fourier_form_ik[:, iproj] ./ sqrt(basis.model.unit_cell_volume)
            wfn / norm(wfn)  # Normalise but not orthogonalise (Why ?)
        end
    end
end