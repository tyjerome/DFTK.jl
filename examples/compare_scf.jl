#DFTK without sym, qe with all symmetries
using DFTK
using LazyArtifacts
using LinearAlgebra
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, krange_spin
using Plots

scfres_333_15 = DFTK.load_scfres("scfres_feo.jld2")
scfres_555_15 = DFTK.load_scfres("scfres_feo_555.jld2")
scfres_777_15 = DFTK.load_scfres("scfres_feo_lda_777_smearing.jld2")
scfres_999_15 = DFTK.load_scfres("scfres_si_lda_999_smearing.jld2")
scfres_111111_15 = DFTK.load_scfres("scfres_si_lda_111111_smearing.jld2")

energy_3_15 = scfres_333_15.energies.total
energy_5_15 = scfres_555_15.energies.total
energy_7_15 = scfres_777_15.energies.total
energy_9_15 = scfres_999_15.energies.total
energy_11_15 = scfres_111111_15.energies.total

nk = [(2i+1)^3 for i in 1:2]
energies = [energy_3_15, energy_5_15]
#energies_qe = [-17.00321378, -17.04533067, -17.04961103, -17.05023439, -17.05034440]/2
energies_qe = [-551.94137982,  -552.09844888]/2
plot(nk, energies-energies_qe, xaxis =:log)