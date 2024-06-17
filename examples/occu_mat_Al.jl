using Bessels
using DFTK
using LazyArtifacts
using LinearAlgebra
using AtomsIOPython
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

orbital_name = "3P"
indices = DFTK.find_orbital_indices(orbital_name, psp.pswfc_labels)
#indices2 = DFTK.find_orbital_indices(orbital_name, psp2.pswfc_labels)

@time occupation_matrix_1 = DFTK.build_occupation_matrix(scfres, psps[1], scfres.basis, scfres.ψ, indices[2], indices[1], psp_positions[1][1])
#occupation_matrix_2 = DFTK.build_occupation_matrix(scfres, psps[3], scfres.basis, scfres.ψ, indices2[2], indices2[1], psp_positions[3][1])
#occ_spin_up_1 = occupation_matrix_1[1]

#Vhub = DFTK.hubbard_u_potential(4.5, 4.5, scfres, psp, basis, scfres.ψ, "3D", positions[1])