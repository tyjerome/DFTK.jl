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

function build_occupation_matrix(scfres, atom_index, psp, basis, ψ, i::Integer, l::Integer, atom_position)
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

    O = [zeros(Complex{Float64}, 2*l+1, 2*l+1) for _ in 1:basis.model.n_spin_components]

    
    #Start calculating Occupation matrix for each atomic sites
    #O_{σ,m1,m2} = Σ_{k, G_k, band} f_{k, band} * w_{k} |ψ_σ,k,band><proj_m1|proj_m2><ψ_σ,k,band|
    #=
    for each k point, the number of G is defined, and so does the number of orbitals should be changed 
    =#
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        #println(σ, ik)
        qs_cart = Gplusk_vectors_cart(basis, basis.kpoints[ik])
        G_k = size(ψ[ik], 1)
        num_band = size(ψ[ik], 2)
        n_k = length(ψ)

        orbital = atomic_wavefunction(basis, atom_index)

        #println(ortho_orbital[1]'*ortho_orbital[1])
        #ortho_orbital = [ortho_lowdin(orbital_ik) for orbital_ik in orbital]
        #println(orbital[1,1])
        for m1 in -l:l
            for m2 in -l:l

                for band in 1:num_band #1:num_band
                    O[σ][m1+l+1, m2+l+1] += occupation[ik][band] * #YJ try ^2
                    basis.kweights[ik] * ψ[ik][:,band]' * orbital[ik][5+m1+l+1] * #try^2
                    orbital[ik][5+m2+l+1]' * ψ[ik][:,band] #/n_k
                end
                #println(ik, ",",m1,",", m2)
            end
        end
    end
    O
end



function rearrange_columns(orbital::Matrix{T}) where T
    #In DFTK, the orbitals first read are arranged as -l, -l+1, ... -1, 0, +1, ...., l-1, l
    #In QE, it is like 0, +1, -1, ... , l, -l
    #This function changes the order of orbital from DFTK-like to QE-like

    # Get the number of orbitals and ms values
    Gs, ms = size(orbital)

    # Calculate the value of l
    l = (ms - 1) ÷ 2

    # Generate the new order list [0, +1, -1, ..., +l, -l]
    new_order = [0; collect(1:l); collect(-1:-1:-l)]

    # Rearrange the columns of the orbital matrix according to the new order
    rearranged_orbital = orbital[:, [i + l + 1 for i in new_order]]

    return rearranged_orbital
end

function build_proj_matrix(scfres, psp, basis, ψ, i, l, atom_position)
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
        #println(σ, ik)
        qs_cart = Gplusk_vectors_cart(basis, basis.kpoints[ik])
        G_k = size(ψ[ik], 1)
        num_band = size(ψ[ik], 2)
        n_k = length(ψ)

        orbital = compute_atom_fourier(basis, scfres.eigenvalues, scfres.ψ, i, l, psp, atom_position)

        #println(orbital[1,1])
        for m1 in -l:l
            for m2 in -l:l

                for band in 1:num_band #1:num_band
                    sum_over_G_k_1 = zero(ComplexF64)
                    sum_over_G_k_2 = zero(ComplexF64)
                    for G_index in 1:G_k
                        sum_over_G_k_1 += orbital[ik][G_index, m1+l+1]
                        sum_over_G_k_2 += conj(orbital[ik][G_index, m2+l+1])
                    end
                    
                    O[σ][m1+l+1, m2+l+1] += occupation[ik][band] *
                    basis.kweights[ik] * sum_over_G_k_1 * sum_over_G_k_2 #/n_k
                end
                #println(ik, ",",m1,",", m2)
            end
        end
    end
    O
end

function rearrange_complex_numbers(matrices::Vector{Matrix{ComplexF64}})
    # Determine the dimensions of the matrices
    m = size(matrices[1], 2) # number of columns
    G = [size(matrix, 1) for matrix in matrices] # G(k) for each matrix
    
    # Calculate the total number of columns in the rearranged matrix
    total_columns = sum(G) * m
    
    # Initialize the new matrix
    rearranged_matrix = Matrix{ComplexF64}(undef, m, total_columns)
    
    # Initialize the column index
    col_index = 1
    
    # Iterate over each matrix
    for matrix in matrices
        # Iterate over each column of the current matrix
        for col in 1:m
            # Get the complex numbers in the current column
            column_values = matrix[:, col]
            
            # Copy the values into the rearranged matrix
            a = size(matrix,1)
            rearranged_matrix[col, col_index:(col_index+a-1)] .= column_values
            
            # Update the column index
            col_index += a
        end
    end
    
    return rearranged_matrix
end

function undo_rearrange_complex_numbers(rearranged_matrix::Matrix{ComplexF64}, G::Vector{Int})
    # Determine the number of matrices
    k = length(G)
    
    # Determine the number of columns in each matrix
    m = size(rearranged_matrix, 1)
    
    # Initialize the vector of matrices
    matrices = Vector{Matrix{ComplexF64}}(undef, k)
    
    # Initialize the column index
    col_index = 1
    
    # Iterate over each matrix
    for i in 1:k
        # Calculate the number of columns for the current matrix
        num_columns = G[i] * m
        
        # Initialize the current matrix
        matrix = Matrix{ComplexF64}(undef, G[i], m)
        
        # Iterate over each column of the current matrix
        for col in 1:m
            # Extract values from the rearranged matrix and assign them to the current column of the matrix
            matrix[:, col] .= rearranged_matrix[col, col_index:col_index+G[i]-1]
            
            # Update the column index
            col_index += G[i]
        end
        
        # Store the current matrix in the vector of matrices
        matrices[i] = matrix
    end
    
    return matrices
end

function compute_atom_fourier(basis, eigenvalues, ψ, i::Integer, l::Integer, psp, position)
    """
    output: orbital[ik][igk, il]
    ik: indice of kpoints
    igk : indice of Gk_cart
    il : m

    1/V^0.5 * exp(-im * (R * G)) * (-im)^l * ylm_real(l, m, p) *
    """
    atom_fourier = Vector{Matrix}(undef, length(eigenvalues))
    position = DFTK.vector_red_to_cart(basis.model, position)

    for (ik, ψk) in enumerate(ψ)

        Gk_cart = Gplusk_vectors_cart(basis, basis.kpoints[ik])
        fourier_form_k = atom_fourier_form(i, l, psp, Gk_cart)
        atom_shift = [dot(position, Gi) for Gi in Gk_cart]
        atom_fourier_k = exp.(-im * atom_shift) .* fourier_form_k #2pi_no need

        atom_fourier[ik] = atom_fourier_k
    end
    return atom_fourier ./ sqrt(basis.model.unit_cell_volume)
    #projs ./ (basis.model.unit_cell_volume)
end

function compute_amn(
    basis::PlaneWaveBasis, ψ::AbstractVector{<:AbstractMatrix{<:Complex}},
    guess::Function; spin::Integer=1,
)
    kpts = krange_spin(basis, spin)
    ψs = ψ[kpts]  # ψ for the selected spin

    n_kpts = length(kpts)
    n_bands = size(ψs[1], 2)
    # I call once `guess` to get the n_wann, to avoid having to pass `n_wann`
    # as an argument.
    ϕk = guess(basis.kpoints[kpts[1]])
    n_wann = size(ϕk, 2)
    A = Wannier.zeros_gauge(eltype(ψs[1]), n_kpts, n_bands, n_wann)
    println(typeof(A), length(A))

    # G_vectors in reduced coordinates.
    # The dot product is computed in the Fourier space.
    for (ik, kpt) in enumerate(basis.kpoints[kpts])
        ψk = ψs[ik]
        ik != 1 && (ϕk = guess(kpt))
        size(ϕk) == (size(ψk, 1), n_wann) || error(
            "ik=$(ik), guess function returns wrong size $(size(ϕk)) != $((size(ψk, 1), n_wann))")
        A[ik] .= ψk' * ϕk
    end
    A
end

function guess_amn_psp(basis::PlaneWaveBasis)
    model = basis.model

    # keep only pseudopotential atoms and positions
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]
    psps = [model.atoms[first(group)].psp for group in psp_groups]
    psp_positions = [model.positions[group] for group in psp_groups]

    isempty(psp_groups) && error("No pseudopotential atoms found in the model.")
    guess(kpt) = build_projection_vectors_pswfcs(basis, kpt, psps, psp_positions)

    # Lowdin orthonormalization
    ortho_lowdin ∘ guess
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

function u_potential_single(u::Float64, l::Integer, occ, proj_mat)
    V_mat = 0.5 * proj_mat + occ
    V = sum(V_mat)
    return u * V
end

function hubbard_u_potential(u1::Float64, u2::Float64, scfres, psp, basis, ψ, orbital_str, atom_position)
    indices = find_orbital_indices(orbital_str, psp.pswfc_labels)
    i = indices[2]
    l = indices[1]
    occ = build_occupation_matrix(scfres, psp, basis, ψ, i, l, atom_position)
    proj_mat = build_proj_matrix(scfres, psp, basis, ψ, i, l, atom_position)
    Vhub_1 = u_potential_single(u1, l, occ[1], proj_mat[1])
    Vhub_2 = u_potential_single(u2, l, occ[2], proj_mat[2])
    Vhub = Vhub_1 + Vhub_2
    return real(Vhub)
end
#=
function hubbard_u_potential(u1::Integer, scfres, psp, basis, ψ, orbital_str::String, atom_position)
    u2 = u1
    indices = find_orbital_indices(orbital_str, psp.pswfc_labels)
    i = indices[2]
    l = indices[1]
    occ = build_occupation_matrix(scfres, psp, basis, ψ, i, l, atom_position)
    proj_mat = build_proj_matrix(scfres, psp, basis, ψ, i, l, atom_position)
    Vhub_1 = u_potential_single(u1, l, occ[1], proj_mat[1])
    Vhub_2 = u_potential_single(u2, l, occ[2], proj_mat[2])
    Vhub = Vhub_1 + Vhub_2
    return Vhub
end
=#
