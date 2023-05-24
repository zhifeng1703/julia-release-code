function parent_path(path; n = 1)
    for i = 1:n 
        path = dirname(path) 
    end
    return path;
end

JULIA_REPO_PATH = parent_path(@__FILE__; n = 2)
JULIA_MANIFOLD_PATH = joinpath(JULIA_REPO_PATH, "manifold")
JULIA_INCLUDE_PATH = joinpath(JULIA_REPO_PATH, "inc")
JULIA_LAPACK_PATH = joinpath(JULIA_INCLUDE_PATH, "myLA")


