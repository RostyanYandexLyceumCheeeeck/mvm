import numpy as np

T = type[np.float32] | type[np.float64]


def find_eps_bits(num_type: T) -> tuple[T, int]:
    """:return: machine epsilon and number of bits in the mantissa."""
    one = num_type(1)
    eps = one
    bits = 0

    while num_type(one + eps / num_type(2)) != one:
        eps = num_type(eps / num_type(2))
        bits += 1    
    
    return eps, bits

def find_limits(num_type: T) -> tuple[int, int]:
    """:return: exponent minimum/maximum."""
    val_min, val_max = num_type(1), num_type(1)
    exp_min, exp_max = 0, 0

    while not np.isinf(num_type(val_max * num_type(2))):
        val_max *= num_type(2)
        exp_max += 1
    
    while num_type(val_min / num_type(2)) > 0:
        val_min /= num_type(2)
        exp_min -= 1
        
    return exp_min, exp_max


def answer(num_type: T): 
    eps, bits = find_eps_bits(num_type)
    exp_min, exp_max = find_limits(num_type)
    info = np.finfo(num_type)

    print(f"--- Тип: {num_type.__name__} ---")
    print(f"finded eps:   {eps}")
    print(f"system eps:   {info.eps}")
    print("-" * 10)

    print(f"finded mantissa bits: {bits}")
    print(f"system mantissa bits: {info.nmant}")
    print("-" * 10)

    print(f"finded min exp:  {exp_min}")
    print(f"system min exp:  {info.minexp}")
    print("-" * 10)

    print(f"finded max exp:  {exp_max}")
    print(f"system max exp:  {info.maxexp}")
    print("-" * 10)

    one = num_type(1)
    v1 = one
    v2 = num_type(one + num_type(eps / num_type(2)))
    v3 = num_type(one + eps)
    v4 = num_type(one + eps + num_type(eps / num_type(2)))

    print(f"1) 1               = {v1}")
    print(f"2) 1 + ε/2         = {v2} (== 1: {v2 == v1})")
    print(f"3) 1 + ε           = {v3} (== 1: {v3 == v1})")
    print(f"4) 1 + ε + ε/2     = {v4}")
    print("-" * 40 + "\n")


if __name__ == "__main__":
    answer(np.float32)
    answer(np.float64)
