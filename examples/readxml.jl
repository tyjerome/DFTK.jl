using EzXML
using DFTK
using Plots

xml = readxml("/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/out/si_lda.save/atomic_proj.xml")

xmlroot = root(xml)
struct xmldata
    kpoints::Vector{Any}
    kweights::Vector{Any}
    E_ks::Vector{Any}
    Atomic_wfcs::Vector{Vector{Any}}
end

lengthkpoint = 56

kpoints = []
kweights = []
E_ks = []
projs = []
roughkpoint = xmlroot.firstelement.nextelement.firstelement

for i in 1:lengthkpoint
    global roughkpoint
    kpointstring = split(nodecontent(roughkpoint), "\n")[2:end-1]
    kpointstring = split(kpointstring[1])
    kpoint = map(kpointstring) do line
        parse.(Float64, line)
    end

    kweight = parse.(Float64,roughkpoint["Weight"])

    roughE_k = roughkpoint.nextelement
    E_kstring = split(nodecontent(roughE_k),"\n")[2:end-1]
    E_kstring = append!(split(E_kstring[1]),split(E_kstring[2]))
    E_k = map(E_kstring) do line
        parse.(Float64, line)
    end
    
    rough_proj = roughE_k.nextelement
    proj = []
    rough_atomic_wfc = rough_proj.firstelement
    
    for i in 1:8 #number of kpoints
        atomic_wfc = map(split(nodecontent(rough_atomic_wfc), "\n")[2:end-1]) do line
            ComplexF64(parse.(Float64,split(line))...)
        end
        push!(proj, atomic_wfc)
        rough_atomic_wfc = rough_atomic_wfc.nextelement
    end

    roughkpoint = rough_proj.nextelement
    push!(kpoints, kpoint)
    push!(kweights, kweight)
    push!(E_ks, E_k)
    push!(projs, proj)
end


data = xmldata(kpoints, kweights,E_ks,projs)

function gaussian(x, mean, sigma)
    return exp(-((x - mean)^2) / (2 * sigma^2)) / (sigma * sqrt(2 * π))
end

function gaussian_delta(center, sigma)
    return x -> gaussian(x, center, sigma)
end

x = -1:0.005:1
y1 = zeros(length(x))

for (i,x) in enumerate(x)
    for k in 1:length(data.kpoints)
        for j in 1:length(data.E_ks[1])
            y1[i] += data.kweights[k]/2 * abs2(data.Atomic_wfcs[k][4][j]) * gaussian(x, data.E_ks[k][j], 0.01)
        end
    end
end
x1 = x.*13.6056931230445-ones(length(x))*0.44349278843156481*13.6056931230445
#norm1 = DFTK.simpson(x1,y1)
#y1_normal = y1/norm1
#=
y2 = zeros(201)
for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y2[i] += scfres.basis.kweights[k] * proj_dos_new[k][j,2]*
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.01) * scfres.occupation[k][j]/2
            y2[i] += scfres.basis.kweights[k] * proj_dos_new[k][j,3]* 
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.01) * scfres.occupation[k][j]/2
            y2[i] += scfres.basis.kweights[k] * proj_dos_new[k][j,4]* 
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.01) * scfres.occupation[k][j]/2
        end
    end
end
=#
#norm2 = DFTK.simpson(x2,y2)
#y2 = y2/norm2

integral1 = DFTK.simpson(x1, y1)
plot(x1, y1, xlims= (-13,1), color =:blue, labels = ("integral1, $integral1"))
#plot!(x2, y2, xlims= (-13,1), color =:orange, labels = ("integral2, $integral2"))
vline!([0], color=:red, linestyle=:dash, label="fermi_level")

savefig("qe_si1_p3.png")