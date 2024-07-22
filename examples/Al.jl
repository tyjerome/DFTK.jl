
using DFTK
using LazyArtifacts
using Bessels
using LinearAlgebra
using AtomsIOPython
using Unitful
using UnitfulAtomic
using JLD2


a = 2.02u"angstrom" #Bohr if nothing
lattice = a * [[1 1 0];
               [1 0 1];
               [0 1 1]]

Al = ElementPsp(:Al; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Al.upf"; rcut = 10))
atoms     = [Al]
positions = [zeros(3)];
kgrid = [3,3,3] 
Ecut = 30
magnetic_moments = [1]
model = model_LDA(lattice, atoms, positions; temperature = 0.005, smearing = Smearing.Gaussian(), symmetries = false, magnetic_moments)
basis = PlaneWaveBasis(model; Ecut, kgrid)
ρ0 = guess_density(basis, magnetic_moments)
scfres = self_consistent_field(basis, tol=1e-8; ρ=ρ0);
scfres.energies
save_scfres("scfres_Al.jld2", scfres)