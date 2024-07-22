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

Ni1 = ElementPsp(:Ni; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Ni.upf"; rcut = 10))
Ni2 = ElementPsp(:Ni; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Ni.upf"; rcut = 10))
O = ElementPsp(:O; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/O.upf"; rcut = 10))
atoms     = [Ni1, Ni2, O, O]
positions = [[0.00 0.00 0.00],
             [0.50 0.50 0.50],
             [0.25 0.25 0.25],
             [0.75 0.75 0.75]]

positions = [zeros(3), ones(3)/2, ones(3)/4, 3*ones(3)/4]

#Initial magnetic moments: in QE example, it is 0.67 and -0.67, (makes almost no difference)
magnetic_moments = [4, -4, 0, 0];

model = model_LDA(lattice, atoms, positions; temperature = 0.005, smearing = Smearing.Gaussian(), symmetries = false, magnetic_moments)
basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
ρ0 = guess_density(basis, magnetic_moments)

scfres = self_consistent_field(basis, tol=1e-8; ρ = ρ0, mixing=KerkerDosMixing())
save_scfres("scfres_NiO.jld2", scfres)