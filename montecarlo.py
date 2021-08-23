import sys
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from depot import VerlustTopf, Depot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
number_of_processes = comm.Get_size()
print(number_of_processes)
print('rank ' + str(rank))

rt_mean = 1.08 ** (1 / 365)
# rt_mean = 1.0
total_years = 30
kt_mean = 1.01**(1/365)
transaction_cost = 1
start_capital = 100000
leverage = 2
simulation_runs = 1000
rebalance_period = np.inf
taxRate = 0.20

total_days = 365 * total_years

rt = np.random.laplace(rt_mean, 0.007, (simulation_runs, total_days)).astype(dtype=np.double)
lev_depot_value = np.zeros((simulation_runs, total_days), dtype=np.double)
normal_depot_value = np.zeros((simulation_runs, total_days), dtype=np.double)
rebalance = np.zeros((simulation_runs, total_days), dtype=np.double)
taxes = np.zeros((simulation_runs, total_days), dtype=np.double)
nonlev_depot_value = np.zeros((simulation_runs, total_days), dtype=np.double)
nonlev_taxes = np.zeros((simulation_runs, 1), dtype=np.double)

for j in range(simulation_runs):
    verlustTopf = VerlustTopf()
    lev_depot = Depot(verlustTopf, taxRate)
    normal_depot = Depot(verlustTopf, taxRate)
    lev_depot.purchase(0, start_capital * (leverage -1))
    normal_depot.purchase(0, start_capital * (2 - leverage))

    lev_depot_value[j, 0] = start_capital * (leverage - 1)
    normal_depot_value[j, 0] = start_capital - lev_depot_value[j,0]
    nonlev_depot_value[j, 0] = start_capital
    for i in range(total_days - 1):
        is_rebalance_day = (i + 1) % rebalance_period == 0
        lev_depot.yieldInterest(2*rt[j,i] - kt_mean)
        normal_depot.yieldInterest(rt[j,i])
        if is_rebalance_day:
            rebalance_amount = lev_depot_value[j, 0] - lev_depot.getCurrentValue()
            if rebalance_amount > 0:
                verlustTopfValue = verlustTopf.value
                sellAmount = normal_depot.calculateSellAmount(rebalance_amount + transaction_cost)
                cash = normal_depot.sell(sellAmount)
                taxes[j, i+1] = taxes[j, i] + sellAmount - cash
                lev_depot.purchase(i+1, rebalance_amount)
            else:
                verlustTopfValue = verlustTopf.value
                cash = lev_depot.sell(-rebalance_amount)
                taxes[j, i+1] = taxes[j, i] - rebalance_amount - cash
                normal_depot.purchase(i+1, -rebalance_amount - transaction_cost)

            rebalance[j, i] = rebalance_amount
        else:
            taxes[j, i+1] = taxes[j, i]
            rebalance[j, i] = 0
        normal_depot_value[j, i + 1] = normal_depot.getCurrentValueTaxed()
        lev_depot_value[j, i + 1] = lev_depot.getCurrentValueTaxed()

        nonlev_depot_value[j, i + 1] = nonlev_depot_value[j, i] * rt[j, i]

    taxes[j, -1] += normal_depot.getCurrentTaxes() + lev_depot.getCurrentTaxes()

    nonlev_taxes[j] = (nonlev_depot_value[j, -1] - nonlev_depot_value[j, 0]) * 0.26
    nonlev_depot_value[j, -1] = nonlev_depot_value[j, -1] - nonlev_taxes[j]
    print('simulation_run: {}, rank: {}'.format(j,rank))
    
all_nonlev_depot = None
all_lev_depot = None
all_normal_depot = None
all_rebalance = None
all_taxes = None
all_nonlev_taxes = None

if rank == 0:
    all_nonlev_depot = np.empty(simulation_runs*number_of_processes)
    all_lev_depot = np.empty(simulation_runs*number_of_processes)
    all_normal_depot = np.empty(simulation_runs*number_of_processes)
    all_rebalance = np.empty(simulation_runs*number_of_processes)
    all_taxes = np.empty(simulation_runs*number_of_processes)
    all_nonlev_taxes = np.empty(simulation_runs*number_of_processes)


comm.Gather(np.ascontiguousarray(nonlev_depot_value[:,-1]), all_nonlev_depot, root=0)
comm.Gather(np.ascontiguousarray(lev_depot_value[:,-1]), all_lev_depot, root=0)
comm.Gather(np.ascontiguousarray(normal_depot_value[:,-1]), all_normal_depot, root=0)
comm.Gather(np.ascontiguousarray(rebalance[:,-1]), all_rebalance, root=0)
comm.Gather(np.ascontiguousarray(taxes[:,-1]), all_taxes, root=0)
comm.Gather(np.ascontiguousarray(nonlev_taxes[:]), all_nonlev_taxes, root=0)


if rank == 0:
    assets_lev_with_rebalance = all_normal_depot + all_lev_depot
    assets_nonlev = all_nonlev_depot

    print('levWithRebalanceResult')
    print('number > 1: {}, mean: {}, var: {}, median: {}, max: {}, taxes: {}'.format(
        np.sum(assets_lev_with_rebalance > start_capital) / assets_lev_with_rebalance.size,
        np.mean(assets_lev_with_rebalance), np.var(assets_lev_with_rebalance, ddof=1),
        np.median(assets_lev_with_rebalance), np.max(assets_lev_with_rebalance), np.mean(all_taxes)))
    print('nonLevResult')
    print('number > 1: {}, mean: {}, var: {}, median: {}, max: {}, taxes: {}'.format(
        np.sum(assets_nonlev > start_capital) / assets_nonlev.size,
        np.mean(assets_nonlev),
        np.var(assets_nonlev, ddof=1),
        np.median(assets_nonlev), np.max(assets_nonlev), np.mean(all_nonlev_taxes)))

    plt.hist((assets_lev_with_rebalance - start_capital) / start_capital, bins=100)
    plt.show()
