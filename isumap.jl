# This project is inspired by github:LUK4S-B/IsUMAP.
#@info "Add packages"
#using Pkg; Pkg.add(["FileIO", "Graphs", "MeshIO", "Meshes", "MultivariateStats", "NearestNeighbors", "Plots", "Random", "SparseArrays", "StaticArrays", "StatsBase", "UnicodePlots"])

@info "Load packages"
using FileIO
using Graphs
using MeshIO
using Meshes
using MultivariateStats
using NearestNeighbors
using Plots
using Random; Random.seed!(0);
using SparseArrays
using StaticArrays
using StatsBase
using UnicodePlots

@enum DataScenario random_scenario torus_scenario hemisphere_scenario
scenario = hemisphere_scenario

fuzzy_and_back = true

@info "Load in datapoints"
# Note: A node considers itself a neighbor of distance 0.
high_dim, input_data, N, k_neighbors = if scenario == random_scenario
  # Random points in the cube.
  high_dim = 3
  N = 100
  input_data = [SVector{high_dim, Float64}(rand(high_dim)) for _ in 1:N]
  k_neighbors = 8
  high_dim, input_data, N, k_neighbors
elseif scenario == torus_scenario
  # Random points on the torus, with replacement.
  m = load("./torus.obj")
  high_dim = 3
  input_data =
    [SVector{high_dim, Float64}(p) for p in m.vertex_attributes.position]
  input_data = sample(input_data, 100; replace=false)
  N = length(input_data)
  k_neighbors = 8
  high_dim, input_data, N, k_neighbors
elseif scenario == hemisphere_scenario
  # Random points on the hemisphere.
  high_dim = 3
  N = 1000
  rand1, rand2 = rand(N), rand(N)
  thetas = acos.(rand1)
  phis = 2*pi*rand2
  xs = sin.(thetas) .* cos.(phis)
  ys = sin.(thetas) .* sin.(phis)
  zs = cos.(thetas)
  k_neighbors = 8
  input_data =
    [SVector{high_dim, Float64}(xs[i],ys[i],zs[i]) for i in eachindex(xs)]
  high_dim, input_data, N, k_neighbors
else
  @error "Unspecified scenario"
end

@info "Run KNN"
function run_knn(data)
  # Euclidean distances are assumed by default.
  kd_tree = KDTree(data)
  idxs, dists = knn(kd_tree, data, k_neighbors)
end
idxs, dists = run_knn(input_data)

@info "Normalize distances to neighbors"
function normalize_distances(dists)
  map(dists) do neighborhood_dists
    farthest_neighbor = maximum(neighborhood_dists)
    map(neighborhood_dists) do n
      n / farthest_neighbor
    end
  end
end
M = normalize_distances(dists)

@info "Merge normlized distances into a single matrix"
function merge_normalized_distances(idxs, M)
  R = spzeros(N,N)
  for (i, neighborhood_dists) in enumerate(M)
    for (j, nd) in zip(idxs[i], neighborhood_dists)
      R[i,j] = nd
    end
  end
  R
end
R = merge_normalized_distances(idxs, M)

@info "Convert to a fuzzy simplicial set"
function to_fuzzy_simplicial_set(R)
  T = dropzeros(R)
  for (i,j,v) in zip(findnz(T)...)
    T[i,j] = exp(-v)
  end
  T
end
T = fuzzy_and_back ? to_fuzzy_simplicial_set(R) : identity(R)

@info "Apply a t-conorm"
function t_conorm(T)
  U = T + T'
  for (i,j,v) in zip(findnz(U)...)
    U[i,j] = min(U[i,j], 1.0)
  end
  U
end
U = t_conorm(T)

@info "Convert to a metric space"
function to_metric_space(U)
  D = dropzeros(U)
  for (i,j,v) in zip(findnz(D)...)
    D[i,j] = -log(v)
  end
  dropzeros!(D)
end
D = fuzzy_and_back ? to_metric_space(U) : identity(U)

@info "Find all shortest distances"
function shortest_paths(D)
  floyd_warshall_shortest_paths(Graph(D), D).dists
end
shortest_dists = shortest_paths(D)

# At this point, we finished Isumap, so pass off to any dimension reduction algorithm.
@info "Finished Isumap"

@info "Execute Multi-Dimensional Scaling (MDS)"
shortest_dists[isinf.(shortest_dists)] .= 0.0
mds = fit(MDS, shortest_dists; distances=true, maxoutdim=2)
embedding = predict(mds)

@info "Visualize Output"

# Create an ASCII plot for REPL use and log files.
println(scatterplot(embedding[1,:], embedding[2,:]))

scenario_name = if scenario == random_scenario
  "random"
elseif scenario == torus_scenario
  "torus"
elseif scenario == hemisphere_scenario
  "hemi"
else
  ""
end

# Assign colors according to degree around the circle.
colors = map(input_data) do p
  atan(p[1], p[3])
end
in_data = reduce(hcat, input_data)
sct = scatter3d(in_data[1,:], in_data[2,:], in_data[3,:],
  zcolor=colors, color=:hsv, legend=false, camera=(16,13),
  title="High-dimensional data")
png(sct, "./$(scenario_name)_highdim_k$(k_neighbors).png")
sct = scatter(embedding[1,:], embedding[2,:], zcolor=colors, color=:hsv, legend=false,
  title="Low-dimensional embedding")
png(sct, "./$(scenario_name)_embedding_k$(k_neighbors).png")

# This output was derived by using the Bounded Sum t-conorm on 100 randomly
# sampled points from the torus with k=8 nearest neighbors.
#=
julia> using UnicodePlots

julia> scatterplot(embedding[1,:], embedding[2,:])
      ┌────────────────────────────────────────┐ 
    1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠂⡗⠠⠵⠙⠢⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⡈⢢⠀⠀⠀⢄⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠁⠀⢀⠀⡀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠁⠁⠘⠀⢂⡀⡀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠙⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠄⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⢍⠍⠉⠉⠉⠉⠉⠉⡏⠉⠉⠉⠉⠉⠉⠉⠉⠉⡉⠉⠉⠉⠉⠉⠉⠉⠉⠉│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠒⢈⡀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⡀⣀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠨⡐⠀⠀⠀⠀⠀⡇⠀⠀⠀⡀⡢⠒⠄⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠆⢄⠀⠀⢀⡇⠐⠀⡀⠈⠂⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⡄⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   -2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      └────────────────────────────────────────┘ 
      ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀ 
=#

