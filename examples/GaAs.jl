using DFTK
using LazyArtifacts
using LinearAlgebra
using Unitful
using UnitfulAtomic
using JLD2

a = 2.8750000000u"angstrom" #in angstrom
lattice = a * [[1 1 0];
               [1 0 1];
               [0 1 1]]

Ga = ElementPsp(:Ga; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Ga.upf"; rcut = 10)) #rmax for qe
As = ElementPsp(:As; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/As.upf"; rcut = 10)) #rmax for qe
               

atoms     = [As, Ga]
positions = [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
#positions = [ones(3)/8, -ones(3)/8]

model = model_LDA(lattice, atoms, positions; temperature = 0)
Ecut = 35
kgrid = [3, 3, 3]
fft_size = compute_fft_size(model, Ecut, kgrid; supersampling=2*sqrt(2))
basis = PlaneWaveBasis(model; Ecut, kgrid, fft_size)


ρ0 = guess_density(basis)

scfres = self_consistent_field(basis, tol=1e-10; ρ = ρ0, mixing=KerkerDosMixing())
save_scfres("scfres_GaAs_nosym.jld2", scfres)