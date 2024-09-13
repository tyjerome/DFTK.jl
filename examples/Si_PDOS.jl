using Bessels
using DFTK
using LazyArtifacts
using LinearAlgebra
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, eval_psp_projector_fourier, krange_spin  
using Plots

using Distributions

scfres = DFTK.load_scfres("scfres_si_lda_333_15.jld2")
basis = scfres.basis
model = scfres.basis.model
ψ = scfres.ψ
psp = basis.model.atoms[1]

orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)

ortho_orbital = Vector{Matrix}(undef, length(ψ))
orbital = Vector{Matrix}(undef, length(ψ))

proj_original = Vector{Matrix}(undef, length(ψ))
proj_simple_single_ortho = Vector{Matrix}(undef, length(ψ))
proj_simple_all_ortho = Vector{Matrix}(undef, length(ψ))
proj_BGS_atom_ortho =  Vector{Matrix}(undef, length(ψ))
proj_BGS_orbital_ortho =  Vector{Matrix}(undef, length(ψ))
proj_QR_all_ortho = Vector{Matrix}(undef, length(ψ))
proj_all_kpoint_ortho = Vector{Matrix}(undef, length(ψ))

proj_ortho_p_plus_all_kpoint_ortho = Vector{Matrix}(undef, length(ψ))


function get_list_atomic_wavefunctions(basis, alpha)
    psp_labels = basis.model.atoms[alpha].psp.pswfc_labels
    list_orbitals = []
    for i in 1:length(psp_labels)
        for j in 1:length(psp_labels[i])
            append!(list_orbitals, 2i-1)
        end
    end
    list_orbitals
end

function get_seperated_atomic_wavefunctions(basis, alpha)
    orbital = DFTK.atomic_wavefunction(basis, alpha)
    psp_labels = basis.model.atoms[alpha].psp.pswfc_labels
    list_orbitals = []
    new_orbitals = Vector{Vector}(undef, length(orbital))
    for i in 1:length(psp_labels)
        for j in 1:length(psp_labels[i])
            append!(list_orbitals, 2i-1)
        end
    end
    for k in 1:length(orbital)
        start = 1
        new_orbital_k = []
        orbital_k = orbital[k]
        for l in 1:length(list_orbitals)
            new_orbital_k_l = orbital_k[start:start+list_orbitals[l]-1]
            push!(new_orbital_k, new_orbital_k_l)
            start += l
        end
        new_orbitals[k] = new_orbital_k
    end
    new_orbitals
end

sep_orbital = get_seperated_atomic_wavefunctions(basis, 1)
all_sep_orbital = []
lengthof_orbitals = []
for j in 1:length(sep_orbital[1])
    all_sep_orbital_j = Matrix{Float64}(undef, 0, size(sep_orbital[1][j], 1))
    for (ik, ψk) in enumerate(ψ)
        orbital_ik = sep_orbital[ik]
        orbital_ik_j = orbital_ik[j]#*scfres.basis.kweights[ik]
        all_sep_orbital_j = vcat(all_sep_orbital_j, stack(orbital_ik_j))
        append!(lengthof_orbitals, size(all_orbital1,1))
    end
    all_sep_orbital_j = DFTK.ortho_lowdin(all_sep_orbital_j)
    all_sep_orbital_j = [all_sep_orbital_j[:,i] for i in 1:size(all_sep_orbital_j,2)]
    push!(all_sep_orbital, all_sep_orbital_j)
end
ortho_orbital_1 = DFTK.ortho_lowdin(all_orbital1)
#all_orbital = hcat(ortho_orbital_1, ortho_orbital_2)
starting_row = 1
new_orbital = Vector{Matrix}(undef, length(ψ))
for (ik, ψk) in enumerate(ψ)
    new_orbital[ik] = all_orbital[starting_row:lengthof_orbitals[ik], :]
    starting_row = lengthof_orbitals[ik]+1
    proj_all_kpoint_ortho[ik] = abs2.(ψk' * new_orbital[ik])
end




for (ik, ψk) in enumerate(ψ)
    orbital1_ik = orbital1[ik]
    orbital2_ik = orbital2[ik]
    orbital_ik = append!(orbital1_ik, orbital2_ik)
    ortho_QR_ik = Matrix(qr(stack(orbital_ik)).Q)
    proj_QR_all_ortho[ik] = abs2.(ψk' * ortho_QR_ik)
end

for (ik, ψk) in enumerate(ψ)
    orbital1_ik = orbital1[ik]
    orbital2_ik = orbital2[ik]
    ortho_orbital1_ik = DFTK.ortho_lowdin(stack(orbital1_ik))
    ortho_orbital2_ik = DFTK.ortho_lowdin(stack(orbital2_ik))
    ortho_orbital2_ik -= ortho_orbital1_ik * (ortho_orbital1_ik' * ortho_orbital2_ik)
    orbital_BGS_atom_ortho_ik = hcat(ortho_orbital1_ik, ortho_orbital2_ik)
    proj_BGS_atom_ortho[ik] = abs2.(ψk' * orbital_BGS_atom_ortho_ik)
end
orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)
for (ik, ψk) in enumerate(ψ)
    orbital1_ik = orbital1[ik]
    orbital2_ik = orbital2[ik]
    orbital_s_ik = append!(orbital1_ik[1:1], orbital2_ik[1:1])
    orbital_p_ik = append!(orbital1_ik[2:4], orbital2_ik[2:4])
    ortho_orbitals_ik = DFTK.ortho_lowdin(stack(orbital_s_ik))
    ortho_orbitalp_ik = DFTK.ortho_lowdin(stack(orbital_p_ik))
    ortho_orbitalp_ik -= ortho_orbitals_ik * (ortho_orbitals_ik' * ortho_orbitalp_ik)
    #ortho_orbitals_ik -= ortho_orbitalp_ik * (ortho_orbitalp_ik' * ortho_orbitals_ik)
    orbital_BGS_orbital_ortho_ik = hcat(ortho_orbitals_ik, ortho_orbitalp_ik)
    for i in 1:size(orbital_BGS_orbital_ortho_ik,2)
        orbital_BGS_orbital_ortho_ik[:,i] = orbital_BGS_orbital_ortho_ik[:,i]/(orbital_BGS_orbital_ortho_ik[:,i]' * orbital_BGS_orbital_ortho_ik[:,i])
    end
    proj_BGS_orbital_ortho[ik] = abs2.(ψk' * orbital_BGS_orbital_ortho_ik)
end

orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)

for (ik, ψk) in enumerate(ψ)
    orbital1_ik = orbital1[ik]
    orbital2_ik = orbital2[ik]
    orbital_ik = append!(orbital1_ik, orbital2_ik)
    orbital_ik = stack(orbital_ik)
    proj_original[ik] = abs2.(ψk' * orbital_ik)
end

orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)

for (ik, ψk) in enumerate(ψ)
    orbital1_ik = orbital1[ik]
    orbital2_ik = orbital2[ik]
    orbital_ik = append!(orbital1_ik, orbital2_ik)
    ortho_orbital_ik = DFTK.ortho_lowdin(stack(orbital_ik))
    proj_simple_all_ortho[ik] = abs2.(ψk' * ortho_orbital_ik)
end

orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)

for (ik, ψk) in enumerate(ψ)
    orbital1_ik = orbital1[ik]
    orbital2_ik = orbital2[ik]
    ortho_orbital1_ik = DFTK.ortho_lowdin(stack(orbital1_ik))
    ortho_orbital2_ik = DFTK.ortho_lowdin(stack(orbital2_ik))
    ortho_orbital_ik = hcat(ortho_orbital1_ik, ortho_orbital2_ik)
    proj_simple_single_ortho[ik] = abs2.(ψk' * ortho_orbital_ik)
end

orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)
all_orbital1 = Matrix{Float64}(undef, 0, size(orbital1[1], 1))
all_orbital2 = Matrix{Float64}(undef, 0, size(orbital2[1], 1))
lengthof_orbitals = []
for (ik, ψk) in enumerate(ψ)
    orbital1_ik = orbital1[ik]*scfres.basis.kweights[ik]
    orbital2_ik = orbital2[ik]*scfres.basis.kweights[ik]
    all_orbital1 = vcat(all_orbital1, stack(orbital1_ik))
    all_orbital2 = vcat(all_orbital2, stack(orbital1_ik))
    append!(lengthof_orbitals, size(all_orbital1,1))
end
ortho_orbital_1 = DFTK.ortho_lowdin(all_orbital1)
ortho_orbital_2 = DFTK.ortho_lowdin(all_orbital2)
all_orbital = hcat(ortho_orbital_1, ortho_orbital_2)
starting_row = 1
new_orbital = Vector{Matrix}(undef, length(ψ))
for (ik, ψk) in enumerate(ψ)
    new_orbital[ik] = all_orbital[starting_row:lengthof_orbitals[ik], :]
    starting_row = lengthof_orbitals[ik]+1
    proj_all_kpoint_ortho[ik] = abs2.(ψk' * new_orbital[ik])
end



function gaussian(x, mean, sigma)
    return exp(-((x - mean)^2) / (2 * sigma^2)) / (sigma * sqrt(2 * π))
end

function gaussian_delta(center, sigma)
    return x -> gaussian(x, center, sigma)
end


x = -10:0.01:10
x = x./13.6056931230445
y1 = zeros(length(x))
y2 = zeros(length(x))
y3 = zeros(length(x))
y4 = zeros(length(x))
y5 = zeros(length(x))
y6 = zeros(length(x))
y7 = zeros(length(x))


for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y1[i] += scfres.basis.kweights[k] * 
            (proj_original[k][j,2]+ proj_original[k][j,3]+proj_original[k][j,4] +
             proj_original[k][j,6]+ proj_original[k][j,7]+proj_original[k][j,8]) *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445 * scfres.occupation[k][j]/2
        end
    end
end

x1 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
norm1 = DFTK.simpson(x1, y1)


for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y2[i] += scfres.basis.kweights[k] * 
            (proj_simple_single_ortho[k][j,2]+ 
            proj_simple_single_ortho[k][j,3]+
            proj_simple_single_ortho[k][j,4]+
            proj_simple_single_ortho[k][j,6]+ 
            proj_simple_single_ortho[k][j,7]+
            proj_simple_single_ortho[k][j,8]) *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445 * scfres.occupation[k][j]/2
        end
    end
end

x1 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
norm1 = DFTK.simpson(x1, y2)


for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y3[i] += scfres.basis.kweights[k] * 
            (proj_simple_all_ortho[k][j,2]+ 
            proj_simple_all_ortho[k][j,3]+
            proj_simple_all_ortho[k][j,4]+
            proj_simple_all_ortho[k][j,6]+ 
            proj_simple_all_ortho[k][j,7]+
            proj_simple_all_ortho[k][j,8]) *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445 * scfres.occupation[k][j] *2
        end
    end
end

x1 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
norm1 = DFTK.simpson(x1, y3)
#

for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y4[i] += scfres.basis.kweights[k] * 
            (proj_BGS_atom_ortho[k][j,2]+ 
            proj_BGS_atom_ortho[k][j,3]+
            proj_BGS_atom_ortho[k][j,4]+
            proj_BGS_atom_ortho[k][j,6]+ 
            proj_BGS_atom_ortho[k][j,7]+
            proj_BGS_atom_ortho[k][j,8]) *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445 * scfres.occupation[k][j]
        end
    end
end

for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y6[i] += scfres.basis.kweights[k] * 
            (proj_BGS_orbital_ortho[k][j,3]+ 
            proj_BGS_orbital_ortho[k][j,4]+
            proj_BGS_orbital_ortho[k][j,5]+
            proj_BGS_orbital_ortho[k][j,6]+ 
            proj_BGS_orbital_ortho[k][j,7]+
            proj_BGS_orbital_ortho[k][j,8]) *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445 * scfres.occupation[k][j]
        end
    end
end

x1 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
norm1 = DFTK.simpson(x1, y6)



for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y5[i] += scfres.basis.kweights[k] * 
            (proj_QR_all_ortho[k][j,2]+ 
            proj_QR_all_ortho[k][j,3]+
            proj_QR_all_ortho[k][j,4]+
            proj_QR_all_ortho[k][j,6]+ 
            proj_QR_all_ortho[k][j,7]+
            proj_QR_all_ortho[k][j,8]) *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445 * scfres.occupation[k][j]
        end
    end
end

x1 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
norm1 = DFTK.simpson(x1, y5)

for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y7[i] += 
            (proj_all_kpoint_ortho[k][j,2]+ 
            proj_all_kpoint_ortho[k][j,3]+
            proj_all_kpoint_ortho[k][j,4]+
            proj_all_kpoint_ortho[k][j,6]+ 
            proj_all_kpoint_ortho[k][j,7]+
            proj_all_kpoint_ortho[k][j,8]) *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445 * scfres.occupation[k][j]/2
        end
    end
end

x1 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
norm1 = DFTK.simpson(x1, y5)

orbital1 = DFTK.atomic_wavefunction(basis, 1)
orbital2 = DFTK.atomic_wavefunction(basis, 2)

using DelimitedFiles

data1 = readdlm("/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_atm#1(Si)_wfc#2(p)")[2:end,:]
data2 = readdlm("/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_atm#2(Si)_wfc#2(p)")[2:end,:]
data3 = readdlm("/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_atm#1(Si)_wfc#2(p)_noinv")[2:end,:]
data4 = readdlm("/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_atm#2(Si)_wfc#2(p)_noinv")[2:end,:]

xvalues = data1[:,1] - 6.3424*ones(length(data1[:,1]))

yvalues1 = (data1[:,2:end])
yvalues2 = (data2[:,2:end])

yvaluetot1 = yvalues1[:,1] + yvalues1[:,2] 
yvaluetot2 = yvalues2[:,1] + yvalues2[:,2]
yvaluetot = yvaluetot1 + yvaluetot2
#norm2 = DFTK.simpson(xvalues, yvaluetot1+yvaluetot2)

#xvalues = data3[:,1] - 6.2241*ones(length(data3[:,1]))

#yvalues3 = (data3[:,2:end])
#yvalues4 = (data4[:,2:end])

#yvaluetot3 = yvalues3[:,1] + yvalues3[:,2] 
#yvaluetot4 = yvalues4[:,1] + yvalues4[:,2]
#norm4 = DFTK.simpson(xvalues, yvaluetot3+yvaluetot4)
#plot!(xvalues, yvaluetot1; linestyle=:dash, label="PDOS_QE_atom1")
#plot!(xvalues, yvaluetot2; linestyle=:dash, label="PDOS_QE_atom2")


plot(xvalues, yvaluetot; linewidth = 2, label ="PDOS_QE_nosym_noinv")
#plot!(xvalues, yvaluetot3+yvaluetot4; linewidth = 2, label ="PDOS_QE_noinv")
#plot!(x1, y1, linestyle=:dash, label = "PDOS_DFTK_original")
plot!(x1, y2, xlims = (-13, 1),linestyle=:dash, label = "PDOS_DFTK_single_ortho")
#plot!(x1, y3, xlims = (-13, 1), linestyle=:dash, label = "PDOS_DFTK_all_simple_ortho")
#plot!(x1, y4, xlims = (-13, 1),linestyle=:dash, label = "PDOS_DFTK_BGS_ortho_atoms")
#plot!(x1, y5, xlims = (-13, 1),linestyle=:dash, label = "PDOS_DFTK_QR_ortho_atoms")
#plot!(x1, y6, xlims = (-13, 1),linestyle=:dash, label = "PDOS_DFTK_BGS_ortho_orbitals")
plot!(x1, y7, xlims = (-13, 1),linestyle=:dash, label = "PDOS_DFTK_all_kpoint_ortho_orbitals")


plot!(legend=:topleft)
savefig("PDOS_compare_2p_withsym.png")
