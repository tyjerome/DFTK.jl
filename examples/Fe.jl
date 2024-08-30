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
kgrid = [11, 11, 11] 
Ecut = 15
magnetic_moments = [4];
model = model_LDA(lattice, atoms, positions; temperature = 0.005, smearing = Smearing.Gaussian(), symmetries = false, magnetic_moments)
fft_size = compute_fft_size(model, Ecut, kgrid; supersampling=2)
basis = PlaneWaveBasis(model; Ecut, kgrid, fft_size)
ρ0 = guess_density(basis, magnetic_moments)
scfres = self_consistent_field(basis, tol=1e-8; ρ=ρ0);
scfres.energies
save_scfres("scfres_Fe.jld2", scfres)