from data import Data
import solver

data = Data.generated_with(N=10, K=3, seed=1)
solver.ip_solver_dyn_sec(data)
