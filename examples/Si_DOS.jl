using DFTK
using LazyArtifacts
using LinearAlgebra
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, krange_spin
using Plots

using Distributions

scfres = DFTK.load_scfres("scfres_si_lda.jld2")
basis = scfres.basis
model = scfres.basis.model
ψ = scfres.ψ
psp = basis.model.atoms[1]

orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)

ortho_orbital = Vector{Matrix}(undef, length(ψ))

for (ik, ψk) in enumerate(ψ)
    orbital1_ik = orbital1[ik]
    orbital2_ik = orbital2[ik]
    orbital_ik = append!(orbital1_ik, orbital2_ik)
    ortho_orbital_ik = DFTK.ortho_lowdin(stack(orbital_ik))
    ortho_orbital[ik] = ortho_orbital_ik
end

function gaussian(x, mean, sigma)
    return exp(-((x - mean)^2) / (2 * sigma^2)) / (sigma * sqrt(2 * π))
end

function gaussian_delta(center, sigma)
    return x -> gaussian(x, center, sigma)
end


x = -10:0.01:10
x = x./13.6056931230445
y1 = zeros(length(x))
y2 = zeros(length(x))

for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y1[i] += scfres.basis.kweights[k] * 
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445 * scfres.occupation[k][j]
        end
    end
end

x1 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
norm1 = DFTK.simpson(x1, y1)
x2 = scfres.εF*13.6056931230445 *2

plot(x1, y1; linestyles =:solid, label = "DFTK_DOS")

using DelimitedFiles

data = readdlm("/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_tot")[2:end,:]

xvalues = data[:,1] - 6.2241*ones(length(data[:,1]))

yvalues = (data[:,2:end])

yvaluetot = yvalues[:,1] + yvalues[:,2]
norm2 = DFTK.simpson(xvalues, yvaluetot)
plot!(xvalues, yvaluetot; linestyle=:dash, label="DOS_QE")
savefig("DOS_compare_test.png")