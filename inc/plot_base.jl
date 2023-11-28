# This code provide functions that are useful for plotting.
using PyPlot;

pygui(true)

function get_mesh(ranges)
    n = length(ranges);
    k = [length(ranges[ii]) for ii = 1:n]
    R = [hcat(ranges[ii]) for ii = 1:n]

    msize = prod(k);
    if (msize == 0)
        throw("Error: empty range encountered when in attempt of generating mesh.\n");
    end

    if (n == 2)
        return hcat(vcat(R[1]' .* ones(k[2])...), vcat(ones(k[1])' .* R[2]...));
    end

    msg("Meshed other than 2D not supported yet.\n")

    return nothing;
end

function get_polar_mesh(ranges; c = zeros(length(ranges)))
    # (r, θ) or (r, ϕ, θ)
    n = length(ranges);
    k = [length(ranges[ii]) for ii = 1:n]

    if n == 2
        r = ranges[1];
        θ = ranges[2];
        x_mesh = [r[ii] * cos(θ[jj]) + c[1] for ii = 1:k[1], jj = 1:k[2]];
        y_mesh = [r[ii] * sin(θ[jj]) + c[2] for ii = 1:k[1], jj = 1:k[2]];
        return x_mesh, y_mesh;
    else
        throw("Error! Mesh not yet suppoerted.")
    end
end



