using Sunny
using GLMakie
using LinearAlgebra
using ProgressMeter
using Base.Threads
using DelimitedFiles
using JLD2
#using CairoMakie
using Printf
using LaTeXStrings
#CairoMakie.activate!()

include("Nonperturbative.jl")

# Standard HChain
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

min_sys(sys; dims=2, maxit=10000)
swt, res, LSWT_plot = LSWT(sys, 1, 1, [[0,0,0], [1,0,0]], 400)
display(LSWT_plot)