using Bessels
using DFTK
using LazyArtifacts
using LinearAlgebra
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, eval_psp_projector_fourier, krange_spin  
using Plots

using Distributions

scfres = DFTK.load_scfres("scfres_si_lda_333_15.jld2")
basis = scfres.basis
model = scfres.basis.model
ψ = scfres.ψ
psp = basis.model.atoms[1]

orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)
orbital3 = DFTK.atomic_wavefunction(basis, 3)
orbital4 = DFTK.atomic_wavefunction(basis, 4)

ortho_orbitals = DFTK.compute_ortho_orbitals(basis)

function plot_pdos_single(scfres, atom_index, orbital_index; 
    ε_min = minimum([minimum(scfres.eigenvalues[ik]) 
            for ik = 1:length(scfres.basis.kpoints)])*1, #.1 - 0.2*scfres.εF,
    ε_max = maximum([maximum(scfres.eigenvalues[ik]) 
            for ik = 1:length(scfres.basis.kpoints)])*1, #.1 - 0.2*scfres.εF,
    ε_ticks = 0.001, basis = scfres.basis, eigenvalues = scfres.eigenvalues, smearing=basis.model.smearing,
    temperature=basis.model.temperature)

    if (temperature == 0) || smearing isa Smearing.None
        error("plot_dos only supports finite temperature")
    end


    orbitals = [DFTK.atomic_wavefunction(basis, n) for n in 1:length(scfres.basis.model.atoms)]
    filled_occ = DFTK.filled_occupation(basis.model)
    ε_list = ε_min:ε_ticks:ε_max
    D = [Vector{Vector}(undef, length(ε_list)) for _ in 1:basis.model.n_spin_components]
    proj = Vector{Vector}(undef, length(scfres.ψ))

    for (ik, ψk) in enumerate(scfres.ψ)

        orbital_k = DFTK.ortho_lowdin(stack(orbitals[atom_index][ik]))
        orbital_ik = [orbital_k[:,i] for i in 1:size(orbital_k,2)]
        orbital_ik = orbitals[atom_index][ik][orbital_index]
        proj[ik] = abs2.(ψk' * orbital_ik)
    end

    D = [Vector(undef, length(ε_list)) for _ in 1:basis.model.n_spin_components]
    for σ = 1:basis.model.n_spin_components
        D_σ = zeros(length(ε_list))
        for (iε, ε) in enumerate(ε_list)
            for ik = krange_spin(basis, σ)
                for (iband, εnk) in enumerate(eigenvalues[ik])
                    enred = (εnk - ε) / temperature
                    D_σ[iε] -= (filled_occ * basis.kweights[ik] / temperature
                    * DFTK.Smearing.occupation_derivative(smearing, enred)) * proj[ik][iband]


                    
                end
            end
        end
    D[σ] = D_σ
    end
(;ε_list, D)
end


Fe_PDOS_d1 = plot_pdos_single(scfres, 1, 2; smearing = DFTK.Smearing.Gaussian(), temperature = 0.005)
Fe_PDOS_d7 = plot_pdos_single(scfres, 1, 3; smearing = DFTK.Smearing.Gaussian(), temperature = 0.005)
Fe_PDOS_d8 = plot_pdos_single(scfres, 1, 4; smearing = DFTK.Smearing.Gaussian(), temperature = 0.005)
Fe_PDOS_d9 = plot_pdos_single(scfres, 1, 9)
Fe_PDOS_d10 = plot_pdos_single(scfres, 1, 10)

D_spinup = Fe_PDOS_d1.D[1] .+ Fe_PDOS_d7.D[1].+ Fe_PDOS_d8.D[1]
D_spinup = Fe_PDOS_d6.D[2] + Fe_PDOS_d7.D[2]+ Fe_PDOS_d8.D[2] + Fe_PDOS_d9.D[2] + Fe_PDOS_d10.D[2]

plot(Fe_PDOS_d1.ε_list.*13.6056931230445, D_spinup)

using DelimitedFiles

data1 = readdlm("/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Fe.pdos_atm#1(Fe1)_wfc#3(d)")[2:end,:]
xvalues = data1[:,1]- 15.5.*ones(length(data1[:,1]))

yvalues1 = (data1[:,2:end]) 
yvaluetot1 = yvalues1[:,1] + yvalues1[:,2] 
plot!(xvalues, yvaluetot1)