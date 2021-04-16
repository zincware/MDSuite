from math import ceil

def compute_dt_all_gases():
    # computes the delta t for all the gases in fs.

    Mweight = {
        "Argon": 39.948,
        "Krypton": 83.8,
        "Xenon": 131.29,
        "CH4": 16.043
    }

    # sigma in Angstroms
    sigma = {
        "Argon": 3.3952,
        "Krypton": 3.6274,
        "Xenon": 3.9011,
        "CH4": 3.7281
    }

    # epsilon divided by kb in K
    eps_div_kb = {
        "Argon": 116.79,
        "Krypton": 162.58,
        "Xenon": 277.55,
        "CH4": 148.55
    }

    kb = 1.38064852 * 10 ** (-23)

    #######

    Na = 6.0221409e+23
    eps = {k: v * kb for k, v in eps_div_kb.items()}
    weight = {k: v / 1000 / Na for k, v in Mweight.items()}
    sigma_SI = {k: v * 10 ** (-10) for k, v in Mweight.items()}

    delta_t_SI = {}
    for gas in eps.keys():
        delta_t_SI[gas] = 0.001 / (((eps[gas] / weight[gas]) ** 0.5) / sigma_SI[gas])

    delta_t_fs = {k: ceil(v * 10 ** 15) for k, v in delta_t_SI.items()}
    return delta_t_fs


def compute_dt_gas(gas_name):
    # get the value for the specific gas, even though we recompute it, it is very cheap.

    delta_t_fs = compute_dt_all_gases()
    return delta_t_fs[gas_name]