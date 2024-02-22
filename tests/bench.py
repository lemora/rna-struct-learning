import jax
import jax.numpy as jnp
from jax import value_and_grad
import json
import numpy as np
from timeit import default_timer as timer

from rnasl.folding_primitives.semiring import make_logsumexp_semiring
from rnasl.folding.nussinov_pf_jax import (
    calc_partition_function,
    calc_base_pair_probs,
    compute_marginal_probs,
    compute_mea_structure, )

n = 170
key = jax.random.PRNGKey(0)
dummy_seq = jax.random.randint(key, shape=(n,), minval=0, maxval=4)
dummy_energies = jnp.zeros((4, 4))
semiring = make_logsumexp_semiring()
h = 3


def benchmark_fn(name, fn, with_grad=False, repeat=10, mode="repeat", results=None):
    fn = jax.jit(fn)
    if with_grad:
        fn = value_and_grad(fn)

    # warmup
    for _ in range(3):
        result = fn(dummy_energies) if with_grad else fn()
        jax.block_until_ready(result)

    entry = {"name": name, "mode": mode, "repeat": repeat}

    if mode == "simple":
        start = timer()
        result = fn(dummy_energies) if with_grad else fn()
        jax.block_until_ready(result)
        duration = timer() - start
        entry["time"] = duration
        print(f"{name:<30} | Time: {duration:.6f}s")

    else:
        times = []
        for _ in range(repeat):
            start = timer()
            result = fn(dummy_energies) if with_grad else fn()
            jax.block_until_ready(result)
            times.append(timer() - start)
        entry.update({
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
        })
        print(f"{name:<30} | Mean: {entry['mean']:.6f}s | Std: {entry['std']:.6f}s")

    if results is not None:
        results.append(entry)

    return result


def run_partition():
    return calc_partition_function(dummy_seq, dummy_energies, semiring, h)


def run_probs():
    Z, Z_p = calc_partition_function(dummy_seq, dummy_energies, semiring, h)
    return calc_base_pair_probs(Z, Z_p, dummy_seq, semiring)


def run_marginals():
    Z, Z_p = calc_partition_function(dummy_seq, dummy_energies, semiring, h)
    P = calc_base_pair_probs(Z, Z_p, dummy_seq, semiring)
    return compute_marginal_probs(P, paired=False)


def run_mea():
    Z, Z_p = calc_partition_function(dummy_seq, dummy_energies, semiring, h)
    P = calc_base_pair_probs(Z, Z_p, dummy_seq, semiring)
    return compute_mea_structure(P)


def run_partition_loss(energies):
    Z, Z_p = calc_partition_function(dummy_seq, energies, semiring, h)
    return jnp.sum(Z) + jnp.sum(Z_p)


def run_probs_loss(energies):
    Z, Z_p = calc_partition_function(dummy_seq, energies, semiring, h)
    P = calc_base_pair_probs(Z, Z_p, dummy_seq, semiring)
    return jnp.sum(P)


def run_marginals_loss(energies):
    Z, Z_p = calc_partition_function(dummy_seq, energies, semiring, h)
    P = calc_base_pair_probs(Z, Z_p, dummy_seq, semiring)
    unpaired_marginals = compute_marginal_probs(P, False)
    return jnp.sum(unpaired_marginals)


if __name__ == "__main__":
    print("Benchmarking JAX RNA folding components...\n")
    results = []

    benchmark_fn("Partition function (Z, Z_p)", run_partition, results=results)
    benchmark_fn("Base-pair probabilities", run_probs, results=results)
    benchmark_fn("Unpaired marginals", run_marginals, results=results)
    benchmark_fn("MEA structure", run_mea, results=results)

    benchmark_fn("Grad: partition function", lambda e: run_partition_loss(e), with_grad=True, results=results)
    benchmark_fn("Grad: base-pair probs", lambda e: run_probs_loss(e), with_grad=True, results=results)
    benchmark_fn("Grad: unpaired marginals", lambda e: run_marginals_loss(e), with_grad=True, results=results)

    res_filename = "bench_results.json"
    with open(res_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBench results written to {res_filename}")

    # grad printing stuff
    # graph_pf = jax.make_jaxpr(jax.grad(run_partition_loss))(dummy_energies)
    # print(f"#Ops in pf grad graph: {len(graph_pf.jaxpr.eqns)}", results=results)
    # graph_bpp = jax.make_jaxpr(jax.grad(run_probs_loss))(dummy_energies)
    # print(f"#Ops in bpp grad graph: {len(graph_bpp.jaxpr.eqns)}", results=results)
    #
    # gradval_umarg, grad_umarg = value_and_grad(run_marginals_loss)(dummy_energies)
    # print("Val margs:", gradval_umarg, results=results)
    # print("Gradient margs:", grad_umarg, results=results)
