using Bessels
using DFTK
using LazyArtifacts
using LinearAlgebra
using AtomsIOPython
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, krange_spin

scfres = load_scfres("scfres.jld2")
propertynames(scfres)
basis = scfres.basis
psp = scfres.basis.model.atoms[1].psp
psp_groups = scfres.basis.model.atom_groups
psps = [scfres.basis.model.atoms[group[1]].psp for group in psp_groups]
psp_positions = [scfres.basis.model.positions[group] for group in psp_groups]
#for (psp, positions) in zip(psps, psp_positions)
#here we just calculate the occupation matrix of the first atom site
psp = psps[1]
positions = psp_positions[1]

orbital_name = "3D"
indices = DFTK.find_orbital_indices(orbital_name, psp.pswfc_labels)

occupation_matrix_1 = DFTK.build_occupation_matrix(scfres, psps[1], scfres.basis, scfres.ψ, indices[2], indices[1], psp_positions[1][1])
occupation_matrix_2 = DFTK.build_occupation_matrix(scfres, psps[3], scfres.basis, scfres.ψ, indices[2], indices[1], psp_positions[3][1])
#occ_spin_up_1 = occupation_matrix_1[1]

Vhub = DFTK.hubbard_u_potential(4.5, 4.5, scfres, psp, basis, scfres.ψ, "3D", positions[1])

#=
function compute_atom_fourier_1(basis, eigenvalues, ψ, i::Integer, l::Integer, psp, position)

    atom_fourier = Vector{Matrix}(undef, length(eigenvalues))
    position = DFTK.vector_red_to_cart(basis.model, position)

    for (ik, ψk) in enumerate(ψ)
        Gk_cart = Gplusk_vectors_cart(basis, basis.kpoints[ik])
        fourier_form_k = atom_fourier_form_1(i, l, psp, Gk_cart)
        atom_shift = [dot(position, Gi) for Gi in Gk_cart]
        atom_fourier_k = exp.(-im * atom_shift) .* fourier_form_k
        atom_fourier[ik] = atom_fourier_k
    end
    return
    #projs ./ (basis.model.unit_cell_volume)
end

function atom_fourier_form_1(i::Integer, l::Integer, psp, G_plus_k)
    T = real(Float64)
    #@assert psp isa PspUpf
    # Pre-compute the radial parts of the  pseudo-atomic wavefunctions at unique |p| to speed up
    # the form factor calculation (by a lot). Using a hash map gives O(1) lookup.

    radials = IdDict{T,T}()  # IdDict for Dual compatibility
    for p in G_plus_k
        p_norm = norm(p)
        if !haskey(radials, p_norm)
            radials[p_norm] = eval_psp_pswfc_fourier(psp, i, l, p_norm)
        end
    end

    form_factors = Matrix{Complex{T}}(undef, length(G_plus_k), 2l+1)
    for (ip, p) in enumerate(G_plus_k)
        radials_p = radials[norm(p)]
        count = 1
        for m = -l:l
            # see "Fourier transforms of centered functions" in the docs for the formula
            angular = (-im)^l * ylm_real(l, m, p)
            form_factors[ip, m+l+1] = radials_p * angular
        end
    end
    form_factors
end

function build_occupation_matrix_1(scfres, psp, basis, ψ, i, l, atom_position)
    """
    Occupation matrix for DFT+U implementation
        Inputs: scfres, psudopotential(usp, in order to get the orbitals) wavefunctions, i, l 
        #types?
        Outputs: Occupation matrix ns_{m1,m2} = Σ_{i} f_i <ψ_i|ϕ^I_m1><ϕ^I_m2|ψ_i>
    """
    occupation = scfres.occupation

    O = [zeros(Complex{Float64}, 2*l+1, 2*l+1) for _ in 1:basis.model.n_spin_components]

    
    #Start calculating Occupation matrix for each atomic sites
    #O_{σ,m1,m2} = Σ_{k, G_k, band} f_{k, band} * w_{k} |ψ_σ,k,band><proj_m1|proj_m2><ψ_σ,k,band|
    #=
    for each k point, the number of G is defined, and so does the number of orbitals should be changed 
    =#
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        println(σ, ik)
        #qs_cart = Gplusk_vectors_cart(basis, basis.kpoints[ik])
        #G_k = size(ψ[ik], 1)
        #num_band = size(ψ[ik], 2)
        #=
        orbital = zeros(ComplexF64, 2*l+1, G_k)
        for m in -l:l
            for G_index in 1:G_k
                orbital[m+l+1, G_index] += atom_fourier(n, l, m, qs_cart[G_index], psp)
            end
        end
        =#
        orbital = compute_atom_fourier_1(basis, scfres.eigenvalues, scfres.ψ, i, l, psp, atom_position)

        #println(orbital[1,1])
        for m1 in -l:l
            for m2 in -l:l

                for band in 1:num_band #1:num_band
                    sum_over_G_k = zero(ComplexF64)
                    
                    for G_index in 1:G_k

                        sum_over_G_k += occupation[ik][band] * basis.kweights[ik] * conj(ψ[ik][G_index,band]) *
                        orbital[G_index, m1+l+1] * conj(orbital[G_index, m2+l+1]) * ψ[ik][G_index,band]
                    end
                    
                    O[σ][m1+l+1, m2+l+1] += sum_over_G_k
                end
                println(ik, ",",m1,",", m2)
            end
        end
    end
    O
end

function what(a, b)
    a * b
end
=#