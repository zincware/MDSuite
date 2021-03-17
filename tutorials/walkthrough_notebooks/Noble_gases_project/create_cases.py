from math import ceil
from pathlib import Path

import pandas as pd

from compute_dt import compute_dt_gas


def replace_variable(old_values, new_values, template_file, output_file):
    with open(output_file, "w") as myfile:
        with open(template_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                # print str(line)
                for old_value, new_value in zip(old_values, new_values):
                    line = line.replace(f'{{{old_value}}}',
                                        str(new_value))  # It will replace if line contain 'old_string' by '+new_string'
                # print ret
                myfile.write(line)


########## Properties required
Mweight = {
    "Argon": 39.948,
    "Xenon": 131.29,
}
eps_div_kb = {
    "Argon": 116.79,
    "Xenon": 227.55,
}
kb = 1.38064852 * 10 ** (-23)
Na = 6.02214076 * 10 ** 23
kCal2J = 4184
eps = {k: v * kb * Na / kCal2J for k, v in eps_div_kb.items()}
sigma = {
    "Argon": 3.3952,
    "Xenon": 3.9011,
}

densities = {
    "Argon": [0.942, 1.375],
    "Xenon": [2.243, 2.916],
}

temperatures = {
    "Argon": [135.6, 92.97],
    "Xenon": [229.0, 176.2],
}

timesteps = {
    "Argon": 3,
    "Xenon": 10,
}

##########
# ACTUAL PROGRAM
# CHOOSE THE GASES HERE

gases = ('Argon', 'Xenon')

template_file = Path('template_fluids.lammps')

# iterate for every gas
for gas in gases:

    timestep = timesteps[gas]
    eps_gas = eps[gas]
    MW_gas = Mweight[gas]
    sigma_gas = sigma[gas]
    cutoff = 3 * sigma_gas

    # iterate for every condition
    for density, temperature in zip(densities[gas], temperatures[gas]):
        # this is to make the folder names
        dens_folder_name = str(density)
        temperature_folder_name = str(temperature)
        condition = f'{dens_folder_name}_gcm3_{temperature_folder_name}_K'
        path = Path(f"./{gas}/{condition}")
        path.mkdir(parents=True, exist_ok=True)

        # replace the values in the template
        # call it input.lammps
        variables_to_replace = {'temperature': temperature,
                                'density': density,
                                "Mweight": MW_gas,
                                "cut_off": cutoff,
                                "epsilon": eps_gas,
                                'sigma': sigma_gas,
                                'timestep': timestep}

        output_file = path / 'input.lammps'
        replace_variable(variables_to_replace.keys(), variables_to_replace.values(),
                         template_file=template_file, output_file=output_file)
