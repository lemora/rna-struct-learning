import pytest
import numpy as np
from rnasl.folding_primitives.semiring import Semiring, NumpySumProductSemiring, NumpyMinPlusSemiring, NumpyLogSumExpSemiring


# ---- helper

def test_logsumexp_semiring_divide():
    semiring = NumpyLogSumExpSemiring()
    res = semiring.div(semiring.zero(), semiring.encode(-3))
    assert np.equal(-np.inf, res)
