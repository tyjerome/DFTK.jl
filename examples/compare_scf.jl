#DFTK without sym, qe with all symmetries

using DFTK
using Plots


scfres_333_15 = DFTK.load_scfres("scfres_si_lda_333_smearing.jld2")
scfres_555_15 = DFTK.load_scfres("scfres_si_lda_555_smearing.jld2")
scfres_777_15 = DFTK.load_scfres("scfres_si_lda_777_smearing.jld2")
scfres_999_15 = DFTK.load_scfres("scfres_si_lda_999_smearing.jld2")
scfres_111111_15 = DFTK.load_scfres("scfres_si_lda_111111_smearing.jld2")

energy_3_15 = scfres_333_15.energies.total
energy_5_15 = scfres_555_15.energies.total
energy_7_15 = scfres_777_15.energies.total
energy_9_15 = scfres_999_15.energies.total
energy_11_15 = scfres_111111_15.energies.total

nk = [(2i+1)^3 for i in 1:5]
energies = [energy_3_15, energy_5_15, energy_7_15, energy_9_15, energy_11_15]
#energies_qe = [-17.00321378, -17.04533067, -17.04961103, -17.05023439, -17.05034440]/2
energies_qe = [-17.00321550,  -17.04533261, -17.04961174, -17.05023466, -17.05023466]/2
plot(nk, energies-energies_qe, xaxis =:log)