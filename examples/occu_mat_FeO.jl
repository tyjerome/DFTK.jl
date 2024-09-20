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
scfres = load_scfres("scfres_feo.jld2")
#propertynames(scfres)
basis = scfres.basis
atoms = basis.model.atoms
psp = scfres.basis.model.atoms[1].psp
psp_groups = scfres.basis.model.atom_groups
psps = [scfres.basis.model.atoms[group[1]].psp for group in psp_groups]
psp_positions = [scfres.basis.model.positions[group] for group in psp_groups]
#for (psp, positions) in zip(psps, psp_positions)
#here we just calculate the occupation matrix of the first atom site
psp = psps[1]
positions = psp_positions[1]

orbital = DFTK.atomic_wavefunction(basis, 1)

@time occupation_matrix_1 = DFTK.build_occupation_matrix(scfres, 1, atoms[1].psp, basis, scfres.ψ, "3D")
occupation_matrix_1[1]

@time occupation_matrix_ortho_1 = DFTK.build_occupation_matrix_ortho(scfres, 1, atoms[1].psp, basis, scfres.ψ, "3D")
occupation_matrix_ortho_1.O[1]