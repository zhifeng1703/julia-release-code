# This is a code that implement essential operation on special orthogonal group.

include("./LA_KWARGS.jl");

msg("Loading /inc/myLA/matrix.jl in " * (DEBUG ? "debug" : "normal") * " mode.\n");

isSquare(X) = (size(X)[1] == size(X)[2]);


function isDiagonal(X)
    n = size(X)[1];
    check = 0;
    for ii = 1:n
        for jj = 1:n
            if ii != jj && abs(X[ii, jj]) > check
                check = abs(X[ii, jj]);
            end            
        end
    end
    return isapprox(0, check, atol = LA_ABSTOL_);
end

function isSkewSym(X)
    m, n = size(X);
    if !isSquare(X)
        msg("Matrix is not squared.\n");
        return false;
    end

    if !isapprox(0, norm(X .+ X'), atol = LA_ABSTOL_)
        msg("Matrix is not skew-symmetric.\n");
        return false;
    end

    return true;
end


function isSym(X)
    m, n = size(X);
    if !isSquare(X)
        msg("Matrix is not squared.\n");
        return false;
    end

    if !isapprox(0, norm(X .- X'), atol = LA_ABSTOL_)
        msg("Matrix is not symmetric.\n");
        return false;
    end

    return true;
end


function isOrthonormal(X)
    n, k = size(X);
    XTX = X' * X;
    if !isapprox(norm(XTX .- diagm(ones(k))), 0, atol = LA_ABSTOL_)
        d_msg("Matrix is not orthonormal.\n");
        return false;
    end
    return true;
end

function isSpecOrth(X)
    n = size(X)[1];
    if !isSquare(X)
        d_msg("Matrix is not squared.\n");
        return false;
    end
    
    if !isOrthonormal(X)
        return false;
    end

    if det(X) < 0
        d_msg("Matrix is not on SO group.\n");
        d_msg("Determinant of the matrix is "*string(det(X)) * "\n");
        return false;
    end

    return true;
end


