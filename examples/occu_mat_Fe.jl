using Bessels
using DFTK
using LazyArtifacts
using LinearAlgebra
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, krange_spin

#get the results from the NiO example
scfres = load_scfres("scfres_Fe.jld2")
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

proj_dos_original = DFTK.compute_pdos_projs(basis, scfres.ψ, psp, positions[1])
proj_dos_new = DFTK.compute_pdos_projs_new(basis, scfres.ψ, 1)
proj_dos_new1 = DFTK.compute_pdos_projs_new1(basis, scfres.ψ, 1)
proj_dos_new - proj_dos_original
#Vhub = DFTK.hubbard_u_potential(4.5, 4.5, scfres, psp, basis, scfres.ψ, "3D", positions[1])