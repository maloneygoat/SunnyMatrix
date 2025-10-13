# A file with all necessary functions for NP NLSWT
# You have a Crystal sys object set up using LSWT in sunny.
# Find minimum, plot it and save for inspection.
# set up LSWT for SSF
# Give Linear magnon, along a path in q_space, plot and save.
# Set up NPT with SWT and (Lmagx, Lmagy, Lmagz) choices
# create paths for 1, 2, and 3d and find com.
# calcualte 2 particle energies
# free 2 particle continuum in direction
# 2 particle intesities from continued fraction


# To execute this file, first create a new environment at your favorite directory. Then enter the julia REPL and run the following commands:
# ]
# activate .
# add https://github.com/Hao-Phys/Sunny.jl/tree/non-perturbative2
# add GLMakie
# add ProgressMeter
# add JLD2
# add DelimitedFiles

# For experiments with Matrix elements, follow the same add my branch;
# add https://github.com/maloneygoat/Sunny.jl/tree/Briannon-perturbative2

# Require a system and its bond to be set up.

using StaticArrays
using Sunny

function min_sys(sys::Sunny.System; dims = 2, maxit=10000)
    randomize_spins!(sys)
    minimize_energy!(sys; maxiters=maxit)
    plot_spins(sys; ndims=dims)
end

function LSWT(sys::Sunny.System, α::Int, β::Int, path, n = 400, regularization=1e-5)
    # perform LSWT in sunny, apply a minimization on the big cell, use reshape to smallest mag unit cell
    swt = SpinWaveTheory(sys; regularization=regularization, measure=ssf_custom((q, ssf) -> real(ssf[α, β]), sys))

    pathrun = q_space_path(cryst, path, n)

    res = intensities_bands(swt, pathrun)
    LSWT_plot = plot_intensities(res; units)
    #display(plot_intensities(res; units))

    # reshape magnetic

    return swt, res, LSWT_plot
end

# set path as [H,0,0], [H,H,0] etc, currenty 1
function twopartenergy(npt::Sunny.NonPerturbativeTheory, Hs, path, Lmag::Int)
    qs  = [[Hs[i]*path[1], Hs[i]*path[2], Hs[i]*path[3]] for i in 1:Lmag]
    com_indices = get_reshaped_cartesian_index(npt, qs)

    num_bands = Sunny.nbands(swt)
    num_2ps = Int(binomial(Lmag^2*num_bands+2-1, 2) / Lmag^2)
    E2ps = zeros(num_2ps, Lmag)

    pm = Progress(Lmag; desc="Calculating two-particle energies")
    Threads.@threads for i in 1:Lmag
        E = calculate_two_particle_energies(npt, num_2ps, com_indices[i])
        E2ps[:, i] = E[1:num_2ps]
        next!(pm)
    end
    return E2ps
end

# Get the free two-particle intensities
function free_two_particle(npt::Sunny.NonPerturbativeTheory, α, β, η, ωs, qs, Lmag, w_max, w_min=0)
    
    # Get the free two-particle intensities,
    Sαβ_fr = zeros(Lmag, length(ωs));
    pm = Progress(Lmag; desc="Calculating the free continuum intensities")
    print("Outside free continuum")
    Threads.@threads for i in 1:Lmag
        ret = Sunny.dssf_free_two_particle_continuum_component(swt, qs[i], ωs, η, α, β; atol=1e-5)
        Sαβ_fr[i, :] = ret
        next!(pm)
    end

    #@. ωs = ωs / abs(J)
    return Sαβ_fr
end


function cf_two_particle(npt::Sunny.NonPerturbativeTheory, α, η, ωs, qs, Lmag, k = 0, n_iters = 30, correct = false)
    # k is to fix momentum, e.g k=1 is the 0 mode as julia using 1 indexing.
    # make resolution better by making smaller
    if k != 0
        Sαβ_cf = 0
        Sαβ_cf = Sunny.dssf_continued_fraction(npt, qs[k], ωs, η, n_iters; single_particle_correction=correct)[:, α]

    else
        Sαβ_cf = zeros(Lmag, length(ωs));
        pm = Progress(Lmag; desc="Calculating intensities using continued fraction")
        Threads.@threads for i in 1:Lmag
            Sαβ_cf[i, :] = Sunny.dssf_continued_fraction(npt, qs[i], ωs, η, n_iters; single_particle_correction=correct)[:, α] # component of spin
            next!(pm)
        end
    end

    # Renormalise enrgies
    #@. ωs = ωs / abs(J)
    return Sαβ_cf
end

# Functions originally from tri_uud_aux.jl
function get_reshaped_cartesian_index(npt::Sunny.NonPerturbativeTheory, qs)
    (; clustersize) = npt
    Nu1, Nu2, Nu3 = clustersize
    all_qs = [[i/Nu1, j/Nu2, k/Nu3] for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]
    com_indices = CartesianIndex[]

    for q in qs
        @assert q in all_qs "The momentum is not in the grid."
        q_reshaped = Sunny.to_reshaped_rlu(npt.swt.sys, q)
        for i in 1:3
            (abs(q_reshaped[i]) < 1e-12) && (q_reshaped = setindex(q_reshaped, 0.0, i))
        end
        q_reshaped = mod.(q_reshaped, 1.0)
        qcom_carts_index = findmin(x -> norm(x - q_reshaped), all_qs)[2]
        push!(com_indices, qcom_carts_index)
    end

    return com_indices
end

function calculate_quartic_corrections(npt::Sunny.NonPerturbativeTheory, num_1ps::Int, qcom_index::CartesianIndex{3}; opts...)
    H1p = zeros(ComplexF64, num_1ps, num_1ps)
    Sunny.one_particle_hamiltonian!(H1p, npt, qcom_index; opts...)
    @assert Sunny.diffnorm2(H1p, H1p') < 1e-10
    hermitianpart!(H1p)
    return H1p
end

# For two-particle calculations
function generate_renormalized_npt(npt::Sunny.NonPerturbativeTheory; single_particle_correction::Bool=true, opts...)
    (; swt, clustersize, qs, Es, Vps, real_space_quartic_vertices, real_space_cubic_vertices) = npt
    N1, N2, N3 = clustersize
    cart_indices = CartesianIndices((1:N1, 1:N2, 1:N3))
    num_1ps = Sunny.nbands(swt)
    Es′ = similar(Es)
    Es′ .= 0.0
    pm = Progress(N1*N2*N3, desc="Generating the renormalized non-perturbative theory")
    Threads.@threads for cart_index in cart_indices
        H1p = calculate_quartic_corrections(npt, num_1ps, cart_index; single_particle_correction, opts...)
        E, V = eigen(H1p)
        # sort the eigenvalues based on the overlap with the original 1-particle states. In this way, we should avoid the band-cross issue
        order = Int[]
        for i in 1:num_1ps
            max_index = argmax(abs.(V[:, i]))
            push!(order, max_index)
        end
        E[1:num_1ps] = E[order]
        Es′[:, cart_index] = E
        next!(pm)
    end

    npt′ = Sunny.NonPerturbativeTheory(swt, clustersize, qs, Es′, Vps, real_space_quartic_vertices, real_space_cubic_vertices, :tensor)
    return npt′
end

function calculate_two_particle_energies(npt::Sunny.NonPerturbativeTheory, num_2ps::Int, qcom_index::CartesianIndex{3})
    H2p = zeros(ComplexF64, num_2ps, num_2ps)
    Sunny.two_particle_hamiltonian!(H2p, npt, qcom_index)
    @assert Sunny.diffnorm2(H2p, H2p') < 1e-12
    hermitianpart!(H2p)
    E, _ = eigen(H2p)
    return E
end

"""
    calculate_two_particle_intensities(npt::Sunny.NonPerturbativeTheory, num_2ps::Int, q, qcom_index::CartesianIndex{3}, ωs, η)

Calculate the two-particle intensity using the Lehmann representation. I.e., get the eigenstates from exact diagonalization.
"""
function calculate_two_particle_ed_intensities(npt::Sunny.NonPerturbativeTheory, num_2ps::Int, q, qcom_index::CartesianIndex{3}, ωs, η)
    num_1ps = Sunny.nbands(npt.swt)
    H2p = zeros(ComplexF64, num_2ps, num_2ps)
    Sunny.two_particle_hamiltonian!(H2p, npt, qcom_index)
    @assert Sunny.diffnorm2(H2p, H2p') < 1e-12
    hermitianpart!(H2p)
    E, V = eigen(H2p; sortby = x -> -1/real(x))
    f0 = Sunny.continued_fraction_initial_states(npt, q, qcom_index)
    f0_3 = view(f0, num_1ps+1:num_1ps+num_2ps, 3)
    amps = zeros(num_2ps)
    for i in axes(V, 2)
        amps[i] = abs2(V[:, i] ⋅ f0_3)
    end
    
    ints = zeros(length(ωs))

    for (iω, ω) in enumerate(ωs)
        for j in 1:num_2ps
            ints[iω] += amps[j] * Sunny.lorentzian(ω - E[j], η)
        end
    end

    return ints
end