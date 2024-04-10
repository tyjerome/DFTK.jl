# # Projected dos
# In this example, we'll plot the projected dos of Silicon.
# The dos is projected on the pseudo-atomic wavefunctions, 
# so the projected dos only supports UPF PSP.

using DFTK
using Unitful
using Plots
using LazyArtifacts

## Define the geometry and pseudopotential
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.0];
    [1 0 1.0];
    [1 1 0.0]]
Si = ElementPsp(:Si; psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Si.upf"))
atoms = [Si, Si]
positions = [ones(3) / 8, -ones(3) / 8]

## Run SCF
model = model_LDA(lattice, atoms, positions; temperature=5e-3)
basis = PlaneWaveBasis(model; Ecut=25, kgrid=[12, 12, 12], symmetries_respect_rgrid=true)
scfres = self_consistent_field(basis, tol=1e-8);

## Plot the projected dos for one pseudo-atomic orbiatl
i = 1 # principal quantum number
l = 0 # angular momentum quantum number 
iatom = 1 # i-th atom 
@time plot_pdos(i, l, iatom, scfres; εrange=(-0.3, 0.6))

## Plot the projected dos for all pseudo-atomic orbiatls
iatom = 1 # i-th atom 
@time plot_pdos(iatom, scfres; εrange=(-0.3, 0.6))