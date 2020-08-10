# Database structure

The idea behind the database structure introduced here is to collect all the data from a project into a single, comrpessed, HDF5 database. The image to follow outlines how this data structure should look, using an example project for NaCl at several temperatures. Ideally, the data structur would be extended one tier higher to allow for multiple molten salts as well. This is not pertinent as of yet and is also rather trivial. 

![database architecture](database_design.png)

The dataset here of positions, velocites, and forces are only applicable to this example. In the main src code you will find methods to extract all the available attributes of the data.
