include("stlog_compact_solver.jl")


global STLOG_RESTART_THRESHOLD = 0.1
STLOG_ENABLE_NEARLOG = false
global STLOG_HYBRID_BCH_MAXITER = 6
global STLOG_HYBRID_BCH_ABSTOL = 1e-6

_STLOG_TEST_SOLVER_STOP = terminator(100, 100000, 1e-7, 1e-7)
_STLOG_TEST_NMLS_SET = NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0)


# test_stlog_profile(80, 40, range(0.1, 1.5π, 20);
#     AbsTol=1e-7, MaxIter=2000, MaxTime=10, loops=3, seed=3484,
#     Solver_Stop=_STLOG_TEST_SOLVER_STOP,
#     NMLS_Set=_STLOG_TEST_NMLS_SET)

# test_stlog_profile(40, 20, range(0.1, 1.5π, 20);
#     AbsTol=1e-7, MaxIter=2000, MaxTime=10, loops=3, seed=rand(1:10000),
#     Solver_Stop=_STLOG_TEST_SOLVER_STOP,
#     NMLS_Set=_STLOG_TEST_NMLS_SET)

stlog_simple_test(n, k, scatter_num, seed) = test_stlog_profile(n, k, range(0.1, 1.5π, scatter_num);
    AbsTol=1e-7, MaxIter=2000, MaxTime=10, loops=3, seed=seed,
    Solver_Stop=_STLOG_TEST_SOLVER_STOP,
    NMLS_Set=_STLOG_TEST_NMLS_SET)