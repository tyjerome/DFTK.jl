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
    fourier_form = atomic_centered_function_form_factors(eval_psp_fourier, G_plus_k_all, lmax, n_funs_per_l)
    map(1:length(basis.kpoints)) do ik
        fourier_form_ik = fourier_form[ik]
        structure_factor_ik = exp.(-im .*[dot(pos_cart, Gik) for Gik in G_plus_k_all[ik]])
        wfn = [structure_factor_ik .* fourier_form_ik[:, iproj] ./ sqrt(basis.model.unit_cell_volume) for iproj in 1:size(fourier_form_ik, 2)]
        for iproj in 1:size(wfn)
            wfn[iproj] = wfn[iproj]/ norm(wfn[iproj])  # Normalise but not orthogonalise (Why ?)
        end
        wfn_matrix = hcat(wfn)
        ortho_wfn_matrix = ortho_lowdin(wfn_matrix)
    end
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
    eval_psp_fourier(i, l, p) = eval_psp_pswfc_fourier(psp, i, l, p)
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