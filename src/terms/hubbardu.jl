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

function build_occupation_matrix(scfres, psp, basis, ψ, i::Integer, l::Integer, atom_position)
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
                        sum_over_G_k_1 += conj(ψ[ik][G_index,band]) * orbital[ik][G_index, m1+l+1]
                        sum_over_G_k_2 += conj(orbital[ik][G_index, m2+l+1]) * ψ[ik][G_index,band]
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

function compute_atom_fourier(basis, eigenvalues, ψ, i::Integer, l::Integer, psp, position)
    """
    output: orbital[ik][igk, il]
    ik: indice of kpoints
    igk : indice of Gk_cart
    il : m
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
    F = svd(A)
    return F.U * F.Vt
end
#=
function atom_fourier_form(i::Integer, l::Integer, psp, G_plus_k::AbstractVector{Vec3{TT}}) where {TT}
    T = real(TT)
    @assert psp isa PspUpf
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
=#
function atom_fourier_form(i::Integer, l::Integer, psp, G_plus_k::AbstractVector{SVector{3, TT}}) where {TT}
    T = real(TT)
    @assert psp isa PspUpf
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

function atom_angular(i::Integer, l::Integer, psp, G_plus_k::AbstractVector{SVector{3, TT}}) where {TT}
    T = real(TT)
    @assert psp isa PspUpf
    # Pre-compute the radial parts of the  pseudo-atomic wavefunctions at unique |p| to speed up
    # the form factor calculation (by a lot). Using a hash map gives O(1) lookup.

    angulars = Matrix{Complex{T}}(undef, length(G_plus_k), 2l+1)
    for (ip, p) in enumerate(G_plus_k)
        for m = -l:l
            # see "Fourier transforms of centered functions" in the docs for the formula
            angular = (-im)^l * ylm_real(l, m, p)
            angulars[ip, m+l+1] = angular
        end
    end
    angulars
end

function build_projection_vectors_pswfcs(basis::PlaneWaveBasis{T}, kpt::Kpoint,
    psps, psp_positions) where {T}
    unit_cell_volume = basis.model.unit_cell_volume
    n_pswfc = count_n_pswfc(psps, psp_positions)
    n_G    = length(G_vectors(basis, kpt))
    proj_vectors = zeros(Complex{T}, n_G, n_pswfc)
    qs = to_cpu(Gplusk_vectors(basis, kpt))

    # Compute the columns of proj_vectors = 1/√Ω pihat(k+G)
    # Since the pi are translates of each others, pihat(k+G) decouples as
    # pihat(q) = ∫ p(r-R) e^{-iqr} dr = e^{-iqR} phat(q).
    # The first term is the structure factor, the second the form factor.
    offset = 0  # offset into proj_vectors
    for (psp, positions) in zip(psps, psp_positions)
        # Compute position-independent form factors
        qs_cart = to_cpu(Gplusk_vectors_cart(basis, kpt))
        form_factors = build_form_factors_pswfcs(psp, qs_cart)

        # Combine with structure factors
        for r in positions
        # k+G in this formula can also be G, this only changes an unimportant phase factor
            structure_factors = map(q -> cis2pi(-dot(q, r)), qs)
            @views for ipswfc = 1:count_n_pswfc(psp)
                proj_vectors[:, offset+ipswfc] .= (
                structure_factors .* form_factors[:, ipswfc] ./ sqrt(unit_cell_volume)
                )
            end
            offset += count_n_pswfc(psp)
        end
    end
    @assert offset == n_pswfc

    # Offload potential values to a device (like a GPU)
    to_device(basis.architecture, proj_vectors)
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
