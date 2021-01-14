""" Calculate the Nernst-Einstein Conductivity of a system """

class NernstEinsteinIonicConductivity:
    """ Class for the calculation of the Nernst-Einstein ionic conductivity """
    def __init__(self):

        raise NotImplementedError
# truth_array = [[bool(self.diffusion_coefficients["Einstein"]["Singular"]),
#                         bool(self.diffusion_coefficients["Einstein"]["Distinct"])],
#                        [bool(self.diffusion_coefficients["Green-Kubo"]["Singular"]),
#                         bool(self.diffusion_coefficients["Green-Kubo"]["Distinct"])]]
#
#         def _ne_conductivity(_diffusion_coefficients):
#             """ Calculate the standard Nernst-Einstein Conductivity for the system
#
#             args:
#                 _diffusion_coefficients (dict) -- dictionary of diffusion coefficients
#             """
#
#             numerator = self.number_of_atoms * (constants.elementary_charge ** 2)
#             denominator = constants.boltzmann_constant * self.temperature * (self.volume * (self.units['length'] ** 3))
#             prefactor = numerator / denominator
#
#             diffusion_array = []
#             for element in self.species:
#                 diffusion_array.append(_diffusion_coefficients["Singular"][element] *
#                                        abs(self.species[element]['charge'][0]) *
#                                        (len(self.species[element]['indices']) / self.number_of_atoms))
#
#             return (prefactor * np.sum(diffusion_array)) / 100
#
#         def _cne_conductivity(_singular_diffusion_coefficients, _distinct_diffusion_coefficients):
#             print("Sorry, this currently isn't available")
#             return
#
#             numerator = self.number_of_atoms * (constants.elementary_charge ** 2)
#             denominator = constants.boltzmann_constant * self.temperature * (self.volume * (self.units['length'] ** 3))
#             prefactor = numerator / denominator
#
#             singular_diffusion_array = []
#             for element in self.species:
#                 singular_diffusion_array.append(_singular_diffusion_coefficients[element] *
#                                                 (len(self.species[element]['indices']) / self.number_of_atoms))
#
#         if all(truth_array[0]) is True and all(truth_array[1]) is True:
#             "Update all NE and CNE cond"
#             pass
#
#         elif not any(truth_array[0]) is True and not any(truth_array[1]) is True:
#             "Run the diffusion analysis and then calc. all"
#             pass
#
#         elif all(truth_array[0]) is True and not any(truth_array[1]) is True:
#             """ Calc NE, CNE for Einstein """
#             pass
#
#         elif all(truth_array[1]) is True and not any(truth_array[0]) is True:
#             """ Calc all NE, CNE for GK """
#             pass
#
#         elif truth_array[0][0] is True and truth_array[1][0] is True:
#             """ Calc just NE for EIN and GK """
#
#             self.ionic_conductivity["Nernst-Einstein"]["Einstein"] = _ne_conductivity(
#                 self.diffusion_coefficients["Einstein"])
#             self.ionic_conductivity["Nernst-Einstein"]["Green-Kubo"] = _ne_conductivity(
#                 self.diffusion_coefficients["Green-Kubo"])
#
#             print(f'Nernst-Einstein Conductivity from Einstein Diffusion: '
#                   f'{self.ionic_conductivity["Nernst-Einstein"]["Einstein"]} S/cm\n'
#                   f'Nernst-Einstein Conductivity from Green-Kubo Diffusion: '
#                   f'{self.ionic_conductivity["Nernst-Einstein"]["Green-Kubo"]} S/cm')
#
#         elif truth_array[0][0] is True and not any(truth_array[1]) is True:
#             """ Calc just NE for EIN """
#
#             self.ionic_conductivity["Nernst-Einstein"]["Einstein"] = _ne_conductivity(
#                 self.diffusion_coefficients["Einstein"])
#             print(f'Nernst-Einstein Conductivity from Einstein Diffusion: '
#                   f'{self.ionic_conductivity["Nernst-Einstein"]["Einstein"]} S/cm')
#
#         elif truth_array[1][0] is True and not any(truth_array[0]) is True:
#             """ Calc just NE for GK """
#
#             self.ionic_conductivity["Nernst-Einstein"]["Green-Kubo"] = _ne_conductivity(
#                 self.diffusion_coefficients["Green-Kubo"])
#             print(f'Nernst-Einstein Conductivity from Green-Kubo Diffusion: '
#                   f'{self.ionic_conductivity["Nernst-Einstein"]["Green-Kubo"]} S/cm')
#
#         elif all(truth_array[0]) is True and truth_array[1][0] is True:
#             """ Calc CNE for EIN and just NE for GK"""
#             pass
#
#         elif all(truth_array[1]) is True and truth_array[0][0] is True:
#             """ Calc CNE for GK and just NE for EIN"""
#             pass
#
#         else:
#             print("This really should not be possible... something has gone horrifically wrong")
#             return