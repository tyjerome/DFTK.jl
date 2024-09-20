using Bessels
using DFTK
using LazyArtifacts
using LinearAlgebra
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, krange_spin
using Plots

using Distributions

#get the results from the NiO example
scfres = DFTK.load_scfres("scfres_si_lda_333_15.jld2")
basis = scfres.basis
psp = scfres.basis.model.atoms[1].psp
psp_groups = scfres.basis.model.atom_groups
psps = [scfres.basis.model.atoms[group[1]].psp for group in psp_groups]
psp_positions = [scfres.basis.model.positions[group] for group in psp_groups]

psp = psps[1]
positions = psp_positions[1]

atoms = basis.model.atoms


#here we just calculate the occupation matrix of the first atom site
#Occupation matrix with all atom orthogonalization
@time occupation_matrix_1 = DFTK.build_occupation_matrix(scfres, 1, atoms[1].psp, basis, scfres.ψ, "3P")
occupation_matrix_1[1]

#Occupation matrix with single atom orthogonalization
@time occupation_matrix_ortho_1 = DFTK.build_occupation_matrix_ortho(scfres, 1, atoms[1].psp, basis, scfres.ψ, "3P")
occupation_matrix_ortho_1.O[1]