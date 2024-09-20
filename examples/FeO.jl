using DFTK
using LazyArtifacts
using LinearAlgebra
using Unitful
using UnitfulAtomic
using JLD2

a = 8.19 #in Bohr
lattice = a / 2 * [[1 1 2];
                   [1 2 1];
                   [2 1 1]]

Fe1 = ElementPsp(:Fe; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Fe.upf"; rcut = 10))
Fe2 = ElementPsp(:Fe; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Fe.upf"; rcut = 10))
O = ElementPsp(:O; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/O.upf"; rcut = 10))
atoms     = [Fe1, Fe2, O, O]
#positions = [[0.00 0.00 0.00],
#             [0.50 0.50 0.50],
#             [0.25 0.25 0.25],
#             [0.75 0.75 0.75]]

positions = [zeros(3), ones(3)/2, ones(3)/4, 3*ones(3)/4]

#Initial magnetic moments: in QE example, it is 0.67 and -0.67, (makes almost no difference)
magnetic_moments = [4, -4, 0, 0];
Ecut = 15
kgrid=[3,3,3]
model = model_LDA(lattice, atoms, positions; temperature = 0.005, symmetries = false, smearing = Smearing.Gaussian(), magnetic_moments)
fft_size = compute_fft_size(model, Ecut, kgrid; supersampling=2*sqrt(2))
basis = PlaneWaveBasis(model; Ecut=15, kgrid, fft_size)
ρ0 = guess_density(basis, magnetic_moments)

scfres = self_consistent_field(basis, tol=1e-8; ρ = ρ0, mixing=KerkerDosMixing())
save_scfres("scfres_feo.jld2", scfres)