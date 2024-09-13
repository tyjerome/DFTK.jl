using Bessels
using DFTK
using LazyArtifacts
using LinearAlgebra
using JLD2
using LinearAlgebra
using Interpolations: linear_interpolation
using DFTK: ylm_real, eval_psp_pswfc_fourier, krange_spin
using Plots

using Distributions

#get the results from the NiO example
scfres = DFTK.load_scfres("scfres_si_lda.jld2")
#scfres = DFTK.unfold_bz(scfres)
#propertynames(scfres)
basis = scfres.basis
psp = scfres.basis.model.atoms[1].psp
psp_groups = scfres.basis.model.atom_groups
psps = [scfres.basis.model.atoms[group[1]].psp for group in psp_groups]
psp_positions = [scfres.basis.model.positions[group] for group in psp_groups]
#for (psp, positions) in zip(psps, psp_positions)
#here we just calculate the occupation matrix of the first atom site
psp = psps[1]
positions = psp_positions[1]

orbital1 = DFTK.atomic_wavefunction(basis, 1)

orbital2 = DFTK.atomic_wavefunction(basis, 2)


for (ik, ψk)in enumerate(scfres.ψ)
    orbital1_ik = orbital1[ik]
    orbital2_ik = orbital2[ik]
end

#@time occupation_matrix_1 = DFTK.build_occupation_matrix(scfres, 1, psp, basis, scfres.ψ, "3D")
occupation_matrix_ortho_1 = DFTK.build_occupation_matrix_ortho(scfres, 1, psp, basis, scfres.ψ, "3S")

proj_dos_new = DFTK.compute_pdos_projs_new(basis, scfres.ψ, 1)

function compute_pdos_projs_ortho(basis, ψ, atom_index) #TODO
    # Build Fourier transform factors centered at 0.
    orbital = DFTK.atomic_wavefunction(basis, atom_index)
    projs = Vector{Matrix}(undef, length(basis.kpoints))
    ortho_orbital = Vector{Matrix}(undef, length(basis.kpoints))
    for (ik, ψk) in enumerate(ψ)
        ortho_orbital_ik = hcat(stack(orbital[ik])[:,1], DFTK.ortho_lowdin(stack(orbital[ik])[:,2:4]))
        ortho_orbital[ik] = ortho_orbtial_ik
        projs[ik] = abs2.(ψk' * ortho_orbital_ik)
    end
   (;projs, ortho_orbital)
end

proj_dos_ortho = DFTK.compute_pdos_projs_ortho(basis, scfres.ψ, 1).projs


proj_dos_overlap = DFTK.compute_pdos_projs_overlap(basis, scfres.ψ, 1)
#Vhub = DFTK.hubbard_u_potential(4.5, 4.5, scfres, psp, basis, scfres.ψ, "3D", positions[1])


# Build Fourier transform factors centered at 0.
fourier_form = DFTK.atomic_wavefunction(basis, 1)
    
projs = Vector{Matrix}(undef, length(basis.kpoints))
for (ik, ψk) in enumerate(scfres.ψ)
    fourier_form_ik = DFTK.ortho_lowdin(stack(fourier_form[ik]))
    projs[ik] = abs2.(ψk' * fourier_form_ik)
end



function gaussian(x, mean, sigma)
    return exp(-((x - mean)^2) / (2 * sigma^2)) / (sigma * sqrt(2 * π))
end

function gaussian_delta(center, sigma)
    return x -> gaussian(x, center, sigma)
end

#=
function pdos(sigma, scfres, proj_dos_new, delta)
    pdosfunction() = nothing
    for k in 1:length(scfres.ψ)
        gauss = proj_dosdelta()
    end
    return
end
=#
x = -10:0.01:10
x = x./13.6056931230445
y1 = zeros(length(x))

for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y1[i] += scfres.basis.kweights[k] * proj_dos_ortho[k][j,1] *
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445 #* scfres.occupation[k][j]/2 
        end
    end
end
x1 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
norm1 = DFTK.simpson(x1,y1)
y1_normal = y1/norm1

y2 = zeros(length(x))
for (i,x) in enumerate(x)
    for k in 1:length(scfres.ψ)
        for j in 1:length(scfres.eigenvalues[1])
            y2[i] += scfres.basis.kweights[k] * proj_dos_ortho[k][j,2]*
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707)  /13.6056931230445
            y2[i] += scfres.basis.kweights[k] * proj_dos_ortho[k][j,3]* 
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445
            y2[i] += scfres.basis.kweights[k] * proj_dos_ortho[k][j,4]* 
                gaussian(x, scfres.eigenvalues[k][j]*2, 0.00707) /13.6056931230445
        end
    end
end

x2 = x.*13.6056931230445-2*ones(length(x))*scfres.εF*13.6056931230445
#norm2 = DFTK.simpson(x2,y2)
#y2 = y2/norm2

integral1 = DFTK.simpson(x1, y1)
integral2 = DFTK.simpson(x2, y2)
plot(x1, y1, xlims= (-13,1), color =:blue, linewidth = 1.5, labels = ("integral1, $integral1"))
plot!(x2, y2, xlims= (-13,1), color =:red, linewidth = 1.5, labels = ("integral2, $integral2"))
a = scfres.εF
vline!([0], color=:red, linestyle=:dash, label="fermi_level")

savefig("DFTK_si1_only_P_ortho_full.png")

using PyPlot
using DelimitedFiles
# Function to read data from a single file
function read_data(filename::String)
    lines = readdlm(filename)

    # Initialize arrays for x values and y values
    x_values = []
    y_values = []

    # Process each line starting from the second line
    for i in 2:size(lines, 1)
        numbers = lines[i, :]
        push!(x_values, numbers[1] - 6.2285)
        push!(y_values, numbers[2] + numbers[3])
    end

    return x_values, y_values
end

# Function to plot data from multiple files
function plot_multiple_files(filenames::Vector{String}, output_filename::String; vline_x::Union{Float64, Nothing}=nothing)
    
    for (i, filename) in enumerate(filenames)
        # Read data from each file
        x_values, y_values = read_data(filename)
        
        # Plot the series
        plot!(x_values, y_values, linestyle =:dash, linewidth = 2, label="File $(i + 1), Series 1")
    end

    # Add vertical dashed red line if specified
    if vline_x !== nothing
        vline(x=vline_x, color="red", linestyle="--", linewidth=1, label="Vertical Line at $vline_x")
    end

    #Add labels, title, and legend (optional)
    xlabel("X Values")
    ylabel("Y Values")
    title("Combined Plot of Multiple Files with Vertical Line")
    legend()

    #Save the figure without showing it
    savefig(output_filename)
end

# List of filenames to read from
filenames = ["/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_atm#1(Si)_wfc#1(s)_nosym_noinv",
             "/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_atm#1(Si)_wfc#2(p)_nosym_noinv"]

# X-coordinate for the vertical dashed red line
vertical_line_x = 0.0  # Replace with your desired x-coordinate

# Plot data from the specified files and save the figure
plot_multiple_files(filenames, "PDOS_compare_only_withsym.png", vline_x=vertical_line_x)