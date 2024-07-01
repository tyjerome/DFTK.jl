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
atoms     = [ElementPsp(:Fe; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Fe.upf"))]
positions = [zeros(3)];
kgrid = [3,3,3] 
Ecut = 30
magnetic_moments = [4];
model = model_LDA(lattice, atoms, positions; temperature = 5e-4, smearing = Smearing.Gaussian(), symmetries = false, magnetic_moments)
basis = PlaneWaveBasis(model; Ecut, kgrid)
ρ0 = guess_density(basis, magnetic_moments)
scfres = self_consistent_field(basis, tol=1e-8; ρ=ρ0);
scfres.energies
save_scfres("scfres_Fe.jld2", scfres)

