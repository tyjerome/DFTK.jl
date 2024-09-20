using DFTK
using LazyArtifacts
using LinearAlgebra
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, krange_spin
using Plots

using Distributions

scfres = DFTK.load_scfres("scfres_si_lda_333_15.jld2")
basis = scfres.basis
model = scfres.basis.model
ψ = scfres.ψ
psp = basis.model.atoms[1]

orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)

ortho_orbital = Vector{Matrix}(undef, length(ψ))

x = -8:0.01:7
x1 = x./13.6056931230445/2
y1 = zeros(length(x1))
for i in 1:length(x1)
    y1[i] = DFTK.compute_dos(x1[i], scfres.basis, scfres.eigenvalues; smearing = DFTK.Smearing.Gaussian(), temperature = 0.005)[1]/13.6056931230445/2
end

function plot_dos(scfres; energy_range=[-10, 10], energy_ticks=0.01, smearing_temperature=0.01)
    x = energy_range[1]:energy_ticks:energy_range[2]
    
end

#x1 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
norm1 = DFTK.simpson(x1, y1)
x2 = scfres.εF*13.6056931230445 *2
x = x .-scfres.εF*13.6056931230445*2
plot(x, y1; linestyles =:solid, label = "DFTK_DOS")

using DelimitedFiles

data = readdlm("/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_tot")[2:end,:]

xvalues = data[:,1] - 6.3924*ones(length(data1[:,1]))

yvalues = (data[:,2:end])

yvaluetot = yvalues[:,1] + yvalues[:,2]
norm2 = DFTK.simpson(xvalues, yvaluetot)
plot!(xvalues, yvaluetot; linestyle=:dash, label="DOS_QE")
title!("Density of States of Si")
xlabel!("Energy(eV)")
savefig("DOS_compare_test.png")