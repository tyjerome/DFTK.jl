using DFTK
using LazyArtifacts
using LinearAlgebra
using Unitful
using UnitfulAtomic
using JLD2
a = 5.42352  # Bohr
lattice = a * [[-1  1  1];
               [ 1 -1  1];
               [ 1  1 -1]]
atoms     = [ElementPsp(:Fe; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Fe.upf"; rcut = 10))]
positions = [zeros(3)];
kgrid = [6, 6, 6] 
Ecut = 20
model = model_LDA(lattice, atoms, positions; temperature = 0.01, smearing = Smearing.Gaussian(), symmetries = false)
basis = PlaneWaveBasis(model; Ecut, kgrid)
ρ0 = guess_density(basis)
scfres = self_consistent_field(basis, tol=1e-6; ρ=ρ0, mixing = KerkerMixing(), damping = 0.5);
scfres.energies
save_scfres("scfres_Fe_nomag_K11_E20_T1e2.jld2", scfres)
