"""
Class for the logging and printing to console.

Summary
-------
Description: This module provides the functionality to write to a logging file and or print to console.
"""

import logging
from datetime import datetime
import os


class Logger:
    """
    Logger Class that hadnles writing ouptut to the console and logfile

    Attributes
    ----------
    logfile_name : str
            The name of the logfile
    logger1 : logging object
            logging object for logging of general files
    logger2 : logging object
            logging object for logging of calculator files
   """

    def __init__(self, logfile_path, logfile_name: str = None):
        """
        Initialise the experiment class.

        Attributes
        ----------
        logfile_name : str
            The name of the logfile
        logger1 : logging object
            logging object for logging of general files
        logger2 : logging object
            logging object for logging of calculator files
        """

        # Taken upon instantiation
        if logfile_name is None:
            logfile_name = datetime.now().replace(microsecond=0).isoformat().replace(':', '-') + ".log"

        logfile = os.path.join(logfile_path, logfile_name)

        # self.logfile_name = logfile_name  # Name of the logfile.

        # set up logging to file - see previous section for more details
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=logfile,
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger().addHandler(console)

        # Now, define a couple of other loggers which might represent areas in your
        # application:

        logger1 = logging.getLogger('general')
        self.logger1 = logger1
        logger2 = logging.getLogger('calculators')
        self.logger2 = logger2

    def __call__(self, context: str = None, level: str = 'INFO', **kwargs):
        # Use switch case in python3.10
        if level == "INFO":
            self.logger1.info(context)


if __name__ == "__main__":
    loggo = Logger('output.log')
    loggo.logfile_name
    loggo.logger1.debug('Quick zephyrs blow, vexing daft Jim.')
    # logger1.info('How quickly daft jumping zebras vex.')
    # logger2.warning('Jail zesty vixen who grabbed pay from quack.')
    # logger2.error('The five boxing wizards jump quickly.')
