"""
NiO example for occupation matrix testing, Lattice and Structure from QE, check QE package q-e/PW/example/example08
antiferromagnetic, non-metallic, upf function, MV smearing 
"""

using DFTK
using LazyArtifacts
using Bessels
using LinearAlgebra
using AtomsIOPython
using JLD2


a = 8.19 #in Bohr
lattice = a / 2 * [[1 1 2];
                   [1 2 1];
                   [2 1 1]]

Ni1 = ElementPsp(:Ni; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Ni.upf"))
Ni2 = ElementPsp(:Ni; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Ni.upf"))
O = ElementPsp(:O; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/O.upf"))
atoms     = [Ni1, Ni2, O, O]
positions = [[0.00 0.00 0.00],
             [0.50 0.50 0.50],
             [0.25 0.25 0.25],
             [0.75 0.75 0.75]]

positions = [zeros(3), ones(3)/2, ones(3)/4, 3*ones(3)/4]

#Initial magnetic moments: in QE example, it is 0.5 and -0.5, (makes almost no difference)
magnetic_moments = [0.5, -0.5, 0, 0];

model = model_LDA(lattice, atoms, positions ; magnetic_moments, temperature=0.01, smearing=Smearing.MarzariVanderbilt())
basis = PlaneWaveBasis(model; Ecut=60, kgrid=[2,2,2])
ρ0 = guess_density(basis, magnetic_moments)

scfres = self_consistent_field(basis, tol=1e-8; ρ = ρ0, mixing=KerkerDosMixing())
save_scfres("scfres.jld2", scfres)