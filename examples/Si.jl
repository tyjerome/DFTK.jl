using DFTK
using LazyArtifacts
using LinearAlgebra
using Unitful
using UnitfulAtomic
using JLD2

a = 2.7154800000u"angstrom" #in angstrom
lattice = a * [[1 1 0];
               [1 0 1];
               [0 1 1]]

Si = ElementPsp(:Si; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Si.upf"; rcut = 10)) #rmax for qe
               

atoms     = [Si, Si]
positions = [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]

model = model_LDA(lattice, atoms, positions; temperature = 0)
Ecut = 15
kgrid = [11, 11, 11]
fft_size = compute_fft_size(model, Ecut, kgrid; supersampling=2*sqrt(2))
basis = PlaneWaveBasis(model; Ecut=15, kgrid=[11, 11, 11], fft_size)


ρ0 = guess_density(basis)

scfres = self_consistent_field(basis, tol=1e-10; ρ = ρ0, mixing=KerkerDosMixing())
save_scfres("scfres_si_lda.jld2", scfres)