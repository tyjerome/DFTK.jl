"""
-------------------------------------------------------------------------------------
implementing Hubbard model to dft (DFT+U)
    Actual addtional inputs : 1) atoms 2) hubbard manifold (n, l) 3) Hubbard parameter according to Hubbard manifold
    Three inputs for calculation : 1) orbitals 2) wavefunctions 3) Hubbard parameter U
    
    The actual calculation process goes like:
    1) get the orbitals
    2) generate orbital occupation matrix n_hat
    3) get V_Hub
    4) update to Kohn Sham potential
------------------------------------------------------------------------------------

"""
#calculate the occupation matrix for certain atom.

function build_occupation_matrix(scfres, atom_index, psp, basis, ψ, orbital_label)
    """
    Occupation matrix for DFT+U implementation
        Inputs: 
        1)scfres, 
        2)psudopotential(usp, in order to get the orbitals), 
        3)basis, 
        4)wavefunctions ψ, 
        5)n : principal quantum number , -> i (check the pswfc_label)
        6)l : angular quantum number,
        7)atom_position : atomic position in the unitcell
        #types?
        Outputs: Occupation matrix ns_{m1,m2} = Σ_{i} f_i <ψ_i|ϕ^I_m1><ϕ^I_m2|ψ_i>
    """
    occupation = scfres.occupation
    l = find_orbital_indices(orbital_label, psp.pswfc_labels)[1]
    O = [zeros(Complex{Float64}, 2*l+1, 2*l+1) for _ in 1:basis.model.n_spin_components]
    count = count_orbital_position(basis, atom_index, orbital_label)
    orbital = atomic_wavefunction(basis, atom_index)
    
    #Start calculating Occupation matrix for each atomic sites
    #O_{σ,m1,m2} = Σ_{k, G_k, band} f_{k, band} * w_{k} |ψ_σ,k,band><proj_m1|proj_m2><ψ_σ,k,band|
    #=
    for each k point, the number of G is defined, and so does the number of orbitals should be changed 
    =#
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        #println(σ, ik)

        num_band = size(ψ[ik], 2)

        for m1 in -l:l
            for m2 in -l:l

                for band in 1:num_band #1:num_band
                    O[σ][m1+l+1, m2+l+1] += occupation[ik][band] *
                    basis.kweights[ik] * ψ[ik][:,band]' * orbital[ik][count+m1+l+1] *
                    orbital[ik][count+m2+l+1]' * ψ[ik][:,band] #/n_k
                end
            end
        end
    end
    O
end

function count_orbital_position(basis, atom_index, orbital_label)
    psp = basis.model.atoms[atom_index].psp
    index = find_orbital_indices(orbital_label, psp.pswfc_labels)
    @assert psp isa PspUpf
    lmax = psp.lmax
    n_funs_per_l = [length(psp.r2_pswfcs[l+1]) for l in 0:lmax]
    #count the number of functions which has smaller l than the specific orbital
    count = 0
    for l = 0:index[1]-1
        n = n_funs_per_l[l+1]
        count += (2l+1)*n
    end
    for n = 1:index[2]-1
        count += (2*index[1]-1)*n
    end
    count
end

function build_proj_matrix(scfres, atom_index, psp, basis, ψ, orbital_label)

    occupation = scfres.occupation
    l = find_orbital_indices(orbital_label, psp.pswfc_labels)[1]
    O = [zeros(Complex{Float64}, 2*l+1, 2*l+1) for _ in 1:basis.model.n_spin_components]
    orbital = atomic_wavefunction(basis, atom_index)
    count = count_orbital_position(basis, atom_index, orbital_label)

    #Start calculating Occupation matrix for each atomic sites
    #O_{σ,m1,m2} = Σ_{k, G_k, band} f_{k, band} * w_{k} |ψ_σ,k,band><proj_m1|proj_m2><ψ_σ,k,band|
    #=
    for each k point, the number of G is defined, and so does the number of orbitals should be changed 
    =#
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        #println(σ, ik)
        G_k = size(ψ[ik], 1)
        num_band = size(ψ[ik], 2)

        #println(orbital[1,1])
        for m1 in -l:l
            for m2 in -l:l

                for band in 1:num_band #1:num_band
                    sum_over_G_k_1 = zero(ComplexF64)
                    sum_over_G_k_2 = zero(ComplexF64)
                    for G_index in 1:G_k
                        sum_over_G_k_1 += orbital[ik][G_index, count+m1+l+1]
                        sum_over_G_k_2 += conj(orbital[ik][G_index, count+m2+l+1])
                    end
                    
                    O[σ][m1+l+1, m2+l+1] += occupation[ik][band] *
                    basis.kweights[ik] * sum_over_G_k_1 * sum_over_G_k_2
                end
            end
        end
    end
    O
end

function build_proj_matrix(scfres, atom_index, psp, basis, ψ, orbital_label)
    """
    Occupation matrix for DFT+U implementation
        Inputs: 
        1)scfres, 
        2)psudopotential(usp, in order to get the orbitals), 
        3)basis, 
        4)wavefunctions ψ, 
        5)n : principal quantum number , -> i (check the pswfc_label)
        6)l : angular quantum number,
        7)atom_position : atomic position in the unitcell
        #types?
        Outputs: Occupation matrix ns_{m1,m2} = Σ_{i} f_i <ψ_i|ϕ^I_m1><ϕ^I_m2|ψ_i>
    """
    occupation = scfres.occupation
    l = find_orbital_indices(orbital_label, psp.pswfc_labels)[1]
    O = [zeros(Complex{Float64}, 2*l+1, 2*l+1) for _ in 1:basis.model.n_spin_components]
    count = count_orbital_position(basis, atom_index, orbital_label)
    
    #Start calculating Occupation matrix for each atomic sites
    #O_{σ,m1,m2} = Σ_{k, G_k, band} f_{k, band} * w_{k} |ψ_σ,k,band><proj_m1|proj_m2><ψ_σ,k,band|
    #=
    for each k point, the number of G is defined, and so does the number of orbitals should be changed 
    =#
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        #println(σ, ik)

        num_band = size(ψ[ik], 2)

        orbital = atomic_wavefunction(basis, atom_index)
        
        for m1 in -l:l
            for m2 in -l:l

                for band in 1:num_band #1:num_band
                    O[σ][m1+l+1, m2+l+1] += occupation[ik][band] *
                    basis.kweights[ik] * ψ[ik][:,band]' * orbital[ik][count+m1+l+1] *
                    orbital[ik][count+m2+l+1]' * ψ[ik][:,band] #/n_k
                end
            end
        end
    end
    O
end

function ortho_lowdin(A::AbstractMatrix)
    #simple Lowdin orthogonalization
    F = svd(A)
    return F.U * F.Vt
end

function find_orbital_indices(element::String, matrix::Vector{Vector{String}})
    """
    helps to read the corresponding r^2 * χ function from r2_pswfcs
    input : 1) name of the orbital (e.g "4D")
            2) psp.pswfc_labels
    output : 2-vector [l, i]
    """
    for (a, row) in enumerate(matrix)
        for (b, val) in enumerate(row)
            if val == element
                return [a-1, b]
            end
        end
    end
    return "Atom does not have this orbital"
end

function u_potential_single(u::Float64, occ, proj_mat)
    V_mat = 0.5 * proj_mat + occ
    V = sum(V_mat)
    return u * V
end

function hubbard_u_potential(u1::Float64, u2::Float64, scfres, atom_index, psp, basis, orbital_label)
    occ = build_occupation_matrix(scfres, atom_index, psp, basis, scfres.ψ, orbital_label)
    proj_mat = build_proj_matrix(scfres, atom_index, psp, basis, scfres.ψ, orbital_label)
    Vhub_1 = u_potential_single(u1, occ[1], proj_mat[1])
    Vhub_2 = u_potential_single(u2, occ[2], proj_mat[2])
    Vhub = Vhub_1 + Vhub_2
    return real(Vhub)
end
#=
struct TermHubbard <: Term
    ops::
end

@timing "ene_ops: hubbardu" function ene_ops(term::TermHubbard, basis::PlaneWaveBasis{T}, ψ, occupation;
    kwargs...) where {T}
    ops = [MagneticFieldOperator(basis, kpoint, term.Apotential)

    E = zero(T)
    E = hubbard_u_potential(u1, u2, scfres, atom_index, psp, basis, ψ, orbital_label, atom_position)
    (; E, ops)
end
=#