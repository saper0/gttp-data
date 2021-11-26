# On Modelling and Solving Green Collaborative Tactical Transportation Planning
This respository stores data about real intermodal transportation infrastructure between major cities, train stations and ports in the Danube Region. The data has been used to produce the results in "On Modelling and Solving Green Collaborative Tactical Transportation Planning".

## doc
This folder contains some documentation on:
i) how the instances (test data) are generated by the instance generator;
ii) cost and emission tables.

Further details can be found in my [master thesis](http://othes.univie.ac.at/67535/) or in our [publication](https://www.pi.events/IPIC2021/sites/default/files/IPIC2021_DRAFT%20PROCEEDINGS_PAPER_POSTER.pdf) at IPIC2021.

## data
This folder contains distance- and time-matrices of road, rail and ship transport connections as well as the coordinates of all used locations in the Danube Region.

## instance-generator
Python instance generator for the GTTP, documentation on usage accessable with python igen.py -h (developed for python 3.8 or higher).

The instances generator needs to be given distances, time and location files as for example given in the *data* directory. For other information on the instance generation process, please see [./doc/data.pdf](https://raw.githubusercontent.com/saper0/gttp-data/main/doc/data.pdf) and the code documentation.

## instances
This folder contains all individual instances (test data) used in "On Modelling and Solving Green Collaborative Tactical Transportation Planning".

## Notes

If you have any questions, please contact Lukas Gosch: *gosch . lukas (at) gmail . com*

## Citation
If you use any of this data in your research, please consider citing our paper. You can find it online (free-access) in the [draft proceedings](https://www.pi.events/IPIC2021/sites/default/files/IPIC2021_DRAFT%20PROCEEDINGS_PAPER_POSTER.pdf) of the IPIC2021. The final conference proceedings of the [IPIC2021](https://www.pi.events/) will soon be made available. 

Lukas Gosch, Matthias Prandtstetter, Karl F. Doerner, On Modelling and Solving Green Collaborative Tactical Transportation Planning, *8th International Physical Internet Conference*, June 15-16, 2021, Athens, Greece

DOI link will be provided once the proceedings are published. 

## Acknowledgements
This work received funding by the Austrian Federal Ministry for Climate Action, Environment, Energy, 
Mobility, Innovation and Technology (BMK) in the research program “Mobilität der Zukunft” under grant 
number 877710 (PhysICAL).
