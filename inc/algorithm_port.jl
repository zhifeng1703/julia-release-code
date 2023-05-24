

mutable struct ARG_MAP
    ARGS::Vector{Any};
    RECS::Vector{Any};
end

struct terminator
    MaxIter
    MaxTime
    AbsTol
    RelTol
    MinStepsize # MSS
    MinAbsUpdate # MAU
    MinRelUpdate # MRU
    MaxSeqBadIt # MSBI
    # NonMonRecord # NMR

    terminator(i, t, a, r; MSS = nothing, MAU = nothing, MRU = nothing, MSBI = 0) = new(i, t, a, r, MSS, MAU, MRU, MSBI)
    terminator(; i=nothing, t=nothing, a=nothing, r=nothing, MSS = nothing, MAU = nothing, MRU = nothing, MSBI = 0) = new(i, t, a, r, MSS, MAU, MRU, MSBI)
end

mutable struct NMLS_Paras
    α_min::Float64
    α_max::Float64
    γ::Float64
    σ::Float64
    M::Int
    NMLS_Paras() = new(0.01, 10.0, 0.1, 0.5, 10)
    NMLS_Paras(α_min, α_max, γ, σ, M) = new(α_min, α_max, γ, σ, M)
end

function check_termination_vec(AbsRec, RelRec, VecRec, TimeRec, StepRec, CurIter, Stop::terminator)
    if !isnothing(AbsRec) && AbsRec[CurIter] < Stop.AbsTol
        return 1;
    end

    if !isnothing(RelRec) && RelRec[CurIter] < Stop.RelTol
        return 2;
    end

    if CurIter >= Stop.MaxIter
        return 3;
    end

    if !isnothing(TimeRec) && TimeRec[CurIter] > Stop.MaxTime
        return 4;
    end

    if Stop.MaxSeqBadIt != 0
        BadItCnt::Int = 0;
        if !isnothing(Stop.MinAbsUpdate)
            for ii = 1:min(Stop.MaxSeqBadIt, CurIter - 1)
                if VecRec[CurIter - ii] < Stop.MinAbsUpdata
                    BadItCnt += 1;
                end
            end
            if BadItCnt >= Stop.MaxSeqBadIt
                return 5;
            end
        end

        BadItCnt = 0;
        if !isnothing(Stop.MinRelUpdate)
            for ii = 1:min(Stop.MaxSeqBadIt, CurIter - 1)
                if (1.0 - AbsRec[CurIter - ii + 1] / AbsRec[CurIter - ii]) < Stop.MinRelUpdata
                    BadItCnt += 1;
                end
            end
            if BadItCnt >= Stop.MaxSeqBadIt
                return 6;
            end
        end

        BadItCnt = 0;
        if !isnothing(Stop.MinStepsize)
            for ii = 1:min(Stop.MaxSeqBadIt, CurIter - 1)
                if StepRec[CurIter - ii] < Stop.MinStepsize
                    BadItCnt += 1;
                end
            end
            if BadItCnt >= Stop.MaxSeqBadIt
                return 7;
            end
        end
    end

    return 0;
end

function check_termination_val(AbsRec, RelRec, VecRec, TimeRec, StepRec, CurIter, Stop::terminator)
    if AbsRec !== nothing && AbsRec < Stop.AbsTol
        return 1;
    end

    if RelRec !== nothing && RelRec < Stop.RelTol
        return 2;
    end

    if CurIter >= Stop.MaxIter
        return 3;
    end

    if TimeRec !== nothing && TimeRec > Stop.MaxTime
        return 4;
    end

    # if Stop.MaxSeqBadIt != 0
    #     BadItCnt::Int = 0;
    #     if !isnothing(Stop.MinAbsUpdate)
    #         for ii = 1:min(Stop.MaxSeqBadIt, CurIter - 1)
    #             if VecRec[CurIter - ii] < Stop.MinAbsUpdata
    #                 BadItCnt += 1;
    #             end
    #         end
    #         if BadItCnt >= Stop.MaxSeqBadIt
    #             return 5;
    #         end
    #     end

    #     BadItCnt = 0;
    #     if !isnothing(Stop.MinRelUpdate)
    #         for ii = 1:min(Stop.MaxSeqBadIt, CurIter - 1)
    #             if (1.0 - AbsRec[CurIter - ii + 1] / AbsRec[CurIter - ii]) < Stop.MinRelUpdata
    #                 BadItCnt += 1;
    #             end
    #         end
    #         if BadItCnt >= Stop.MaxSeqBadIt
    #             return 6;
    #         end
    #     end

    #     BadItCnt = 0;
    #     if !isnothing(Stop.MinStepsize)
    #         for ii = 1:min(Stop.MaxSeqBadIt, CurIter - 1)
    #             if StepRec[CurIter - ii] < Stop.MinStepsize
    #                 BadItCnt += 1;
    #             end
    #         end
    #         if BadItCnt >= Stop.MaxSeqBadIt
    #             return 7;
    #         end
    #     end
    # end

    return 0;
end