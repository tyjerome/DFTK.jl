
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

Al = ElementPsp(:Al; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Al.upf"))
atoms     = [Al]

positions = [zeros(3)]

magnetic_moments = [3];
kgrid=[5,5,5];
Ecut = 60;

model = model_LDA(lattice, atoms, positions ; temperature=0.0073498618000, magnetic_moments,
    smearing=Smearing.MarzariVanderbilt(), symmetries = false)
fft_size = compute_fft_size(model, Ecut, kgrid; supersampling=480)
basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)
ρ0 = guess_density(basis, magnetic_moments)

scfres = self_consistent_field(basis, tol=1e-10; ρ = ρ0, mixing=KerkerDosMixing())
save_scfres("scfres_Al.jld2", scfres)