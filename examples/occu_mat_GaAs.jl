using Bessels
using DFTK
using LazyArtifacts
using LinearAlgebra
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, krange_spin
using Plots
#get the results from the NiO example
scfres = load_scfres("scfres_GaAs.jld2")
#propertynames(scfres)
basis = scfres.basis
psp = scfres.basis.model.atoms[1].psp
psp_groups = scfres.basis.model.atom_groups
psps = [scfres.basis.model.atoms[group[1]].psp for group in psp_groups]
psp_positions = [scfres.basis.model.positions[group] for group in psp_groups]
#for (psp, positions) in zip(psps, psp_positions)
#here we just calculate the occupation matrix of the first atom site
psp = psps[1]
positions = psp_positions[1]

orbital = DFTK.atomic_wavefunction(basis, 1)

#@time occupation_matrix_1 = DFTK.build_occupation_matrix(scfres, 1, psp, basis, scfres.ψ, "3D")
#@time occupation_matrix_ortho_1 = DFTK.build_occupation_matrix_ortho(scfres, 1, psp, basis, scfres.ψ, "3D")

proj_dos_new1 = DFTK.compute_pdos_projs_new(basis, scfres.ψ, 1)
proj_dos_ortho2 = DFTK.compute_pdos_projs_ortho(basis, scfres.ψ, 2)
proj_dos_new2 = DFTK.compute_pdos_projs_new(basis, scfres.ψ, 2)

proj_dos_new_overlap = DFTK.compute_pdos_projs_overlap(basis, scfres.ψ, 1)
#Vhub = DFTK.hubbard_u_potential(4.5, 4.5, scfres, psp, basis, scfres.ψ, "3D", positions[1])

function gaussian(x, mean, sigma)
    return exp(-((x - mean)^2) / (2 * sigma^2)) / (sigma * sqrt(2 * π))
end

function gaussian_delta(center, sigma)
    return x -> gaussian(x, center, sigma)
end

x = -1:0.01:1

y1 = zeros(201)

for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y1[i] += 1/84 * proj_dos_ortho2[k][j,2] *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.01)*scfres.occupation[k][j]/2/13.6056931230445
            y1[i] += 1/84 * proj_dos_ortho2[k][j,3] * 
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.01)*scfres.occupation[k][j]/2/13.6056931230445
            y1[i] +=  1/84 * proj_dos_ortho2[k][j,4] * 
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.01)*scfres.occupation[k][j]/2/13.6056931230445
        end
    end
end
x1 = x.*13.6056931230445-scfres.εF*2 * 13.6056931230445*ones(201)

plot(x1, y1, xlims= (-16,1), xticks = -16:2:0, color =:blue, label = "Ga_4P")

y2 = zeros(201)
for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y2[i] +=  1/84 * proj_dos_ortho2[k][j,1] *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.01)*scfres.occupation[k][j]/2/13.6056931230445
        end
    end
end
x2 = x.*13.6056931230445-scfres.εF*2 * 13.6056931230445*ones(201)
integral1 = DFTK.simpson(x1, y1)
integral2 = DFTK.simpson(x2, y2)
plot(x1, y1, xlims= (-16,1), xticks = -16:2:0, color =:blue, label = "Ga_4P, integral1, $integral1")
plot!(x2, y2, xlims= (-16,1), xticks = -16:2:0, color =:orange, labels = ("Ga_4P, integral2, $integral2"))

a = scfres.εF
vline!([0], color=:red, linestyle=:dash, label="fermi_level")
savefig("DFTK_GaAs_Ga_wokpoint.png")