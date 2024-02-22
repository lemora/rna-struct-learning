from rnasl.jax_setup import jfloat

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import scipy.special as ssp
from jax import lax

import rnasl.gconst as gc


# ---- interfaces/abstract classes

class Semiring:
    def add(self, a: float, b: float) -> float:
        raise NotImplementedError

    def add_n(self, terms: list[float]) -> float:
        """Add multiple elements (e.g. min or logsumexp over a list)"""
        raise NotImplementedError

    def mul(self, a: float, b: float) -> float:
        raise NotImplementedError

    def div(self, a: float, b: float) -> float:
        raise NotImplementedError

    def zero(self) -> float:
        raise NotImplementedError

    def one(self) -> float:
        raise NotImplementedError

    def equal(self, a: float, b: float):
        raise NotImplementedError

    def to_real_matrix(self, P: list[list[float]]):
        return P

    def encode(self, energy: float) -> float:
        """Convert a free energy value (Delta G) into a semiring element."""
        raise NotImplementedError

    def is_zero(self, x: float):
        return x == self.zero()

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


class NumpySemiring(Semiring):
    def equal(self, a: float, b: float):
        return np.isclose(a, b, atol=1e-6)


class JaxSemiring(Semiring):
    def equal(self, a: float, b: float):
        return jnp.isclose(a, b, atol=1e-6)


# --- implementations NumPy


class NumpyMinPlusSemiring(NumpySemiring):
    def add(self, a: float, b: float) -> float:
        return np.minimum(a, b)

    def add_n(self, terms: list[float]) -> float:
        return np.min(terms)

    def mul(self, a: float, b: float) -> float:
        return a + b

    def div(self, a: float, b: float) -> float:
        return a - b

    def zero(self) -> float:
        return np.inf

    def one(self) -> float:
        return 0.0

    def encode(self, energy: float, temperature: float = gc.TEMP) -> float:
        return energy

    def __eq__(self, other):
        return isinstance(other, NumpyMinPlusSemiring)

    def __hash__(self):
        return hash("NumpyMinPlusSemiring")


class NumpyMaxPlusSemiring(NumpySemiring):
    def add(self, a: float, b: float) -> float:
        return np.maximum(a, b)

    def add_n(self, terms: list[float]) -> float:
        return np.max(terms)

    def mul(self, a: float, b: float) -> float:
        return a + b

    def div(self, a: float, b: float) -> float:
        return a - b

    def zero(self) -> float:
        return -np.inf

    def one(self) -> float:
        return 0.0

    def encode(self, energy: float, temperature: float = gc.TEMP) -> float:
        return energy

    def __eq__(self, other):
        return isinstance(other, NumpyMaxPlusSemiring)

    def __hash__(self):
        return hash("NumpyMaxPlusSemiring")


class NumpySumProductSemiring(NumpySemiring):
    def add(self, a: float, b: float) -> float:
        return a + b

    def add_n(self, terms: list[float]) -> float:
        return np.sum(terms)

    def mul(self, a: float, b: float) -> float:
        return a * b

    def div(self, a: float, b: float) -> float:
        return a / b

    def zero(self) -> float:
        return 0.0

    def one(self) -> float:
        return 1.0

    def encode(self, energy: float, temperature: float = gc.TEMP) -> float:
        return np.exp(-energy / (gc.K_B * temperature))

    def __eq__(self, other):
        return isinstance(other, NumpySumProductSemiring)

    def __hash__(self):
        return hash("NumpySumProductSemiring")


class NumpyLogSumExpSemiring(NumpySemiring):
    def add(self, a: float, b: float) -> float:
        return float(ssp.logsumexp(np.array([a, b])))

    def add_n(self, terms: list[float]) -> float:
        return float(ssp.logsumexp(terms))

    def mul(self, a: float, b: float) -> float:
        return a + b

    def div(self, a: float, b: float) -> float:
        return a - b

    def zero(self) -> float:
        return -np.inf

    def one(self) -> float:
        return 0.0

    def to_real_matrix(self, P: list[list[float]]):
        return np.exp(P)

    def encode(self, energy: float, temperature: float = gc.TEMP) -> float:
        return -energy / (gc.K_B * temperature)

    def __eq__(self, other):
        return isinstance(other, NumpyLogSumExpSemiring)

    def __hash__(self):
        return hash("NumpyLogSumExpSemiring")


# --- implementations JAX

class MinPlusSemiring(JaxSemiring):
    def add(self, a: float, b: float) -> jfloat:
        return jnp.minimum(a, b).astype(jfloat)

    def add_n(self, terms: jnp.ndarray) -> jfloat:
        if len(terms) == 0:
            return self.zero()
        return jnp.min(jnp.stack(terms)).astype(jfloat)

    def mul(self, a: jfloat, b: jfloat) -> jfloat:
        return (a + b).astype(jfloat)

    def div(self, a: jfloat, b: jfloat) -> jfloat:
        return (a - b).astype(jfloat)

    def zero(self):
        return jnp.array(-jnp.inf, dtype=jfloat)

    def one(self):
        return jnp.array(0.0, dtype=jfloat)

    def encode(self, energy: jfloat, temperature: jfloat = gc.TEMP) -> jfloat:
        return energy.astype(jfloat)

    def __eq__(self, other):
        return isinstance(other, MinPlusSemiring)

    def __hash__(self):
        return hash("MinPlusSemiring")


class SumProductSemiring(JaxSemiring):
    def add(self, a: jfloat, b: jfloat) -> jfloat:
        return (a + b).astype(jfloat)

    def add_n(self, terms: jnp.ndarray) -> jfloat:
        return jnp.sum(terms).astype(jfloat)

    def mul(self, a: jfloat, b: jfloat) -> jfloat:
        return (a * b).astype(jfloat)

    def div(self, a: jfloat, b: jfloat) -> jfloat:
        return (a / b).astype(jfloat)

    def zero(self) -> jfloat:
        return jnp.array(0.0, dtype=jfloat)

    def one(self) -> jfloat:
        return jnp.array(1.0, dtype=jfloat)

    def encode(self, energy: jfloat, temperature: jfloat = gc.TEMP) -> jfloat:
        return jnp.exp(-energy / (gc.K_B * temperature)).astype(jfloat)

    def __eq__(self, other):
        return isinstance(other, SumProductSemiring)

    def __hash__(self):
        return hash("SumProductSemiring")


class LogSumExpSemiring(JaxSemiring):
    def add(self, a: jfloat, b: jfloat) -> jfloat:
        return jnp.logaddexp(a, b).astype(jfloat)

    def add_n(self, terms: jnp.ndarray) -> jfloat:
        return lax.cond(
            terms.size == 0,
            lambda _: self.zero(),
            lambda _: jsp.logsumexp(terms).astype(jfloat),
            operand=None
        )

    def mul(self, a: jfloat, b: jfloat) -> jfloat:
        return (a + b).astype(jfloat)

    def div(self, a: jfloat, b: jfloat) -> jfloat:
        return (a - b).astype(jfloat)

    def zero(self):
        return jnp.array(-jnp.inf, dtype=jfloat)

    def one(self):
        return jnp.array(0.0, dtype=jfloat)

    def to_real_matrix(self, P: jnp.ndarray):
        return jnp.exp(P)

    def encode(self, energy: jfloat, temperature: jfloat = gc.TEMP) -> jfloat:
        return (-energy / (gc.K_B * temperature)).astype(jfloat)

    def __eq__(self, other):
        return isinstance(other, LogSumExpSemiring)

    def __hash__(self):
        return hash("LogSumExpSemiring")


@dataclass(frozen=True)
class JaxSemiringFrozen:
    add: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    add_n: Callable[[jnp.ndarray], jnp.ndarray]
    mul: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    div: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    zero: Callable[[], jnp.ndarray]
    one: Callable[[], jnp.ndarray]
    encode: Callable[[jnp.ndarray], jnp.ndarray]
    to_real_matrix: Callable[[jnp.ndarray], jnp.ndarray]
    is_zero: Callable[[jnp.ndarray], jnp.ndarray]


def make_logsumexp_semiring(temperature: float = gc.TEMP) -> JaxSemiringFrozen:
    kB = gc.K_B

    def add(a, b):
        a = jnp.nan_to_num(a, nan=zero())
        b = jnp.nan_to_num(b, nan=zero())
        return jnp.logaddexp(a, b)

    def add_n(terms):
        return lax.cond(
            (terms.size == 0) | jnp.all(is_zero(terms)),
            lambda _: zero(),
            lambda _: jsp.logsumexp(terms).astype(jfloat),
            operand=None
        )

    def mul(a, b):
        return a + b

    def div(a, b):
        a = jnp.nan_to_num(a, nan=-jnp.inf)
        b = jnp.nan_to_num(b, nan=-jnp.inf)
        safe = ~(jnp.isneginf(a) & jnp.isneginf(b))
        result = a - b
        result = jnp.where(safe, result, -jnp.inf)
        return result

    def zero():
        return jnp.array(-jnp.inf, dtype=jfloat)  # log(0)

    def one():
        return jnp.array(0.0, dtype=jfloat)  # log(1)

    def is_zero(x):
        return jnp.isneginf(x)

    def encode(energy):
        return (-energy / (kB * temperature)).astype(jfloat)

    def to_real_matrix(P):
        return jnp.exp(P).astype(jfloat)

    return JaxSemiringFrozen(
        add=add,
        add_n=add_n,
        mul=mul,
        div=div,
        zero=zero,
        one=one,
        encode=encode,
        to_real_matrix=to_real_matrix,
        is_zero=is_zero,
    )
