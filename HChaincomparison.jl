using Revise
using Sunny
using GLMakie
using LinearAlgebra
using ProgressMeter
using Base.Threads
using DelimitedFiles
using JLD2
using CairoMakie
using Printf
using LaTeXStrings
CairoMakie.activate!()

# Standard HChain
Lmag = 57
δω = 0.5
wmax = 5
ωs = collect(0:δω:wmax) # create array of energy poins, energies we calculate DSSF at S(q,ω).
η  = 2δω
ν  = 0.0005
cluster = (Lmag, 1, 1)
cartcluster = CartesianIndex(cluster)

# Path
Hs  = [i/Lmag for i in 0:Lmag-1]
qs  = [[Hs[i], 0, 0] for i in 1:Lmag]
units = Units(:meV, :angstrom)
a = 3.0
b = 8.0
c = 8.0
latvecs = lattice_vectors(a, b, c, 90, 90, 90)
positions = [[0, 0, 0]]
cryst = Crystal(latvecs, positions)

sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
J = -1
set_exchange!(sys, J, Bond(1, 1, [1, 0, 0]))

Sunny.min_sys(sys; dims=2, maxit=10000)
swt, res, LSWT_plot = Sunny.LSWT(sys, cryst, 1, 1, [[0,0,0], [1,0,0]], 400)
display(LSWT_plot)
# NPT
npt = Sunny.NonPerturbativeTheory(swt, (Lmag, 1, 1))
Sxx_graph = Sunny.cf_two_particle(npt, 1, η, ωs, qs; k = 0, n_iters = 10000)

cf_fig = Figure(resolution = (1200, 600))
Label(cf_fig[0, 1:3], latexstring("J = $J"), fontsize = 24, halign = :center, valign = :center)
ax_l = Axis(cf_fig[1, 1];  xlabel="[H, 0, 0]", ylabel="Energy (meV)", yticks=collect(0:0.5:wmax) ,title =latexstring("\\text{NLSWT}"))
ylims!(ax_l, 0, wmax)
scale = ReversibleScale(x -> asinh(x / 2) / log(10), x -> 2sinh(log(10) * x))
hm = heatmap!(ax_l, Hs, ωs, Sxx_graph; colorscale = scale, colormap = :heat)
colorbar = Colorbar(cf_fig[1, 2], hm; label="Intensity", width=15)
ax_r = plot_intensities!(cf_fig[1, 3], res; title = latexstring("\\text{LSWT}"))
ax_r.ylabel = "Energy (meV)"
ax_r.yticks = collect(0:0.5:wmax)
ylims!(ax_r, 0, wmax)
display(cf_fig)

# q must be a vector. Result is a 15 x 3 matrix, x3 as x,y,z. 15 as all 0 to wps
f0 = Sunny.continued_fraction_initial_states(npt, [0,0,0], CartesianIndex(Lmag, 1, 1))

# Now we try and get the linear response, lets do x
n_states = size(f0, 1)

abssquaredf0x = abs2.(f0[:, 1]) # first term in the Linear response

# Next we need to find the energies of each of these states.
# then we can do a heatmap plot to find out where they lie. 

E, V = Sunny.truncated_hilbert_space_eigen(npt, [0.5,0,0])

E
V
length(E)

# Now we wanna calculate S_{q=0}
function delta_approx(x, x0; ν=0.05)
    return ν / (π * ((x - x0)^2 + ν^2))
end


function χ_1(ωs, abssquaredf0x, E; ν=0.0005)
    χ_vals = zeros(length(ωs))
    for (i, ω) in enumerate(ωs)
        for n in 1:length(E)
            χ_vals[i] += abssquaredf0x[n] * delta_approx(ω, E[n]; ν=ν)
        end
    end
    return χ_vals
end

x_vals = χ_1(ωs, abssquaredf0x, E, ν=0.05)


# Loops

qs_vals = [q[1] for q in qs]

χ_matrix = zeros(length(ωs), length(qs))

for (i,q) in enumerate(qs)
    f0 = 0
    f0 = Sunny.continued_fraction_initial_states(npt, q, CartesianIndex(Lmag, 1, 1))
    E, V = Sunny.truncated_hilbert_space_eigen(npt, q)
    # Sort everything
    sorted_inds = sortperm(E)
    E_sorted = E[sorted_inds]
    V_sorted = V[:, sorted_inds]

    # Renromalise everything
    f0_vec = f0[:, 1]
    f0_vec ./= norm(f0_vec)
    abssquaredf0x = abs2.(f0_vec' * V_sorted)
    abssquaredf0x ./= sum(abssquaredf0x)
    x_vals = χ_1(ωs, abssquaredf0x, E; ν=0.4)
    x_vals ./= sum(x_vals) 
    χ_matrix[:, i] .= x_vals
end

tr_matrix = transpose(χ_matrix)


# Mistake is assuming En and Fn are in the same order
χ_fig = Figure(resolution = (1200, 600))
ax = Axis(χ_fig[1, 1]; xlabel="[H, 0, 0]", ylabel="Energy (meV)", yticks=collect(0:0.5:wmax) ,title =latexstring("χ_1 \\text{ for Heisenberg Chain}"))
ax_m = Axis(χ_fig[1, 3];  xlabel="[H, 0, 0]", ylabel="Energy (meV)", yticks=collect(0:0.5:wmax) ,title =latexstring("\\text{NLSWT}"))
ylims!(ax, 0, wmax)
scale = ReversibleScale(x -> asinh(x / 2) / log(10), x -> 2sinh(log(10) * x))
xhm = heatmap!(ax, qs_vals, ωs, tr_matrix ; colorscale=scale, interpolate = false, colormap = :heat)
colorbar = Colorbar(χ_fig[1, 2], xhm; label="Intensity", width=15)
hm = heatmap!(ax_m, Hs, ωs, Sxx_graph; colorscale = scale, colormap = :heat)
colorbar = Colorbar(χ_fig[1, 4], hm; label="Intensity", width=15)
ax_r = plot_intensities!(χ_fig[1, 5], res; title = latexstring("\\text{LSWT}"))
ax_r.ylabel = "Energy (meV)"
ax_r.yticks = collect(0:0.5:wmax)
ylims!(ax_r, 0, wmax)
display(χ_fig)

#CairoMakie.save("χ1_HChain_Lmag$Lmag.png", χ_fig)