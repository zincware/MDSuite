from os import path
from pathlib import Path

import GPUtil
import scooby


class Report(scooby.Report):
    """A class for custom scooby.Report."""

    def __init__(self, additional=None, ncol=3, text_width=80, sort=False,
                 gpu=True):
        """Generate a :class:`scooby.Report` instance.

        Parameters
        ----------
        additional : list(ModuleType), list(str)
            List of packages or package names to add to output information.

        ncol : int, optional
            Number of package-columns in html table; only has effect if
            ``mode='HTML'`` or ``mode='html'``. Defaults to 3.

        text_width : int, optional
            The text width for non-HTML display modes

        sort : bool, optional
            Alphabetically sort the packages

        gpu : bool
            Gather information about the GPU. Defaults to ``True`` but if
            experiencing renderinng issues, pass ``False`` to safely generate
            a report.

        """
        # Mandatory packages.
        here = path.abspath(path.dirname(__file__))
        main_mdsuite_folder = Path(path.dirname(path.dirname(here)))  # go two folders up
        requirements_file = main_mdsuite_folder / 'requirements.txt'

        with open(requirements_file) as requirements_file:
            # Parse requirements.txt, ignoring any commented-out lines.
            requirements = [line.split('~')[0] for line in requirements_file.read().splitlines()
                            if not line.startswith('#')]

        # add yaml and remove PyYAML...the package is called in one way and then it is imported in another...
        requirements.append('yaml')
        requirements.remove('PyYAML')
        try:
            gpus = GPUtil.getGPUs()

            list_gpus = []
            for gpu in gpus:
                # get the GPU id
                gpu_id = gpu.id
                # name of GPU
                gpu_name = gpu.name
                # get total memory
                gpu_total_memory = f"{gpu.memoryTotal}MB"
                list_gpus.append(('GPU ID', gpu_id))
                list_gpus.append(('GPU Name', gpu_name))
                list_gpus.append(('GPU Total Memory', gpu_total_memory))
        except:
            list_gpus = [('GPU NAME', 'No Info Available')]

        scooby.Report.__init__(self, additional=additional, core=requirements,
                               ncol=ncol,
                               text_width=text_width, sort=sort,
                               extra_meta=list_gpus)
