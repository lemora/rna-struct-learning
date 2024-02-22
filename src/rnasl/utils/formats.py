import jax.numpy as jnp

# ------ seq to/from int encoding

BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}

CANONICAL_PAIRS = {
    (BASE_TO_INT['C'], BASE_TO_INT['G']),
    (BASE_TO_INT['A'], BASE_TO_INT['U']),
    (BASE_TO_INT['G'], BASE_TO_INT['U']),
}

CANONICAL_MASK = jnp.zeros((4, 4), dtype=bool)
for i, j in CANONICAL_PAIRS:
    CANONICAL_MASK = CANONICAL_MASK.at[i, j].set(True)
    CANONICAL_MASK = CANONICAL_MASK.at[j, i].set(True)

NONCANONICAL_MASK = ~CANONICAL_MASK


def encode_seq(seq: str, base_to_int: dict = BASE_TO_INT) -> list[int]:
    return [base_to_int[base.upper()] for base in seq]


def decode_seq(encoded_seq: list, int_to_base: dict = INT_TO_BASE) -> str:
    return ''.join(int_to_base[i] for i in encoded_seq)


JAX_BASE_ENCODER = jnp.full((256,), -1, dtype=jnp.int32).at[
    jnp.array([ord(k) for k in BASE_TO_INT.keys()])
].set(jnp.array(list(BASE_TO_INT.values()), dtype=jnp.int32))


def encode_seq_jax(seq: str) -> jnp.ndarray:
    seq = seq.upper()
    byte_seq = jnp.array([ord(c) for c in seq], dtype=jnp.uint8)
    return JAX_BASE_ENCODER[byte_seq]


def decode_seq_jax(encoded_seq: jnp.ndarray) -> str:
    decoded = [INT_TO_BASE[int(x)] for x in encoded_seq]
    return ''.join(decoded)


def is_canonical(a: str, b: str):
    ai = BASE_TO_INT[a]
    bi = BASE_TO_INT[b]
    return (ai, bi) in CANONICAL_PAIRS or (bi, ai) in CANONICAL_PAIRS


def is_canonical_jax(a: int, b: int) -> jnp.bool_:
    return CANONICAL_MASK[a, b]


# ------ structure to/from vienna/base pairs

def pairing_to_vienna(sequence: str, pairs: list[tuple[int, int]]) -> str:
    """Generate vienna notation from a list of paired bases (index pairs)"""
    dot_bracket = ['.' for _ in range(len(sequence))]
    for i, j in pairs:
        if i == j:
            continue
        dot_bracket[min(i, j)] = '('
        dot_bracket[max(i, j)] = ')'
    return ''.join(dot_bracket)


def vienna_to_pairing(vienna: str) -> list[tuple[int, int]]:
    """Generate list of paired bases from vienna notation string"""
    stack = []
    pairs = []
    for i, symbol in enumerate(vienna):
        if symbol == '(':
            stack.append(i)
        elif symbol == ')':
            if not stack:
                raise ValueError(f"Unmatched closing parenthesis at position {i}")
            j = stack.pop()
            pairs.append((j, i))
    if stack:
        raise ValueError(f"Unmatched opening parentheses at positions: {stack}")
    return sorted(pairs)
