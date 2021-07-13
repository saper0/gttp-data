# Introduction

This folder contains the locations as well as distance and time matrices used for this work. The data is partly based upon test instances released [here](https://plis.univie.ac.at/research/a-matheuristic-for-a-multimodal-long-haul-routing-problem/) by Wolfinger et al. [1]. The following sections describe the data files and how they differ from Wolfinger et al. [1] and how they got extended. Depending on what data you are interested in, you may want to consider citing Wolfinger et al. [1].

[1] David Wolfinger, Fabien Tricoire, and Karl F. Doerner. *A matheuristic for a multimodal long haul routing problem.* In: EURO Journal on Transportation and Logistics Volume 8, Issue 4, 2019, pp.397–433.

## locations.txt

This file follows the following format:

ID;ID';Type;Latitude;Longitude;Name;Size

There are three types of locations differentiated:
1. Random addresses in city regions
2. Ports
3. Train Stations

ID is a unique identifyer for each location in the file. Wolfinger et al. [1] gave only one location for each city region, so for this work, we have generated nine additional random addresses for each city region. ID' refers to the original ID used by Wolfinger et al. [1]. Furthermore, we have added a size column indicating if the industrial city has a population P of < 100.000 (S), 100.000 <= P < 1.000.000 (M) or P >= 1.000.000 (L).

## distances.txt

The first two header rows indicate, which rows correspond to which transport mode. It reads as follows: The first 376.382 data rows correspond to road distances for lorries (L), the next 72 data rows to distances for ships (S), i.e. distances of waterways between ports and the last 2970 data rows concern rail route distances between train stations (T).

A data row has the following format:
ID1;ID2;Distance [km]

ID1 and ID2 denote the IDs of the connected locations from locations.txt. A connection is assumed to be bidirectional.

Distances of waterways and distances of rail routes have been taken from Wolfinger et al. [1]. Road distances have been recalculated using Ariadne [2].

[2] Matthias Prandtstetter, Markus Straub, and Jakob Puchinger. *On the way to a multi-modal energy-efficient route.* In: IECON2013-39th Annual Conference of the IEEE Industrial Electronics Society. 2013, pp.4779–4784

## traveltimes.txt

The time-matrix data file follows the same format as *distances.txt*. However, a data row includes in the third column travel times in days instead of distances in km. 

Travel times have been generated from *distances.txt* by assuming for lorries average speeds of 60 km/h and 10h operating time. For rail transportation 18 km/h with 24h operation time is assumed and waterway travel times concern push crafts taken from Wolfinger et al. [1]. Push crafts are the most common vessels on the Danube [3].

[3] Manual on Danube Navigation [https://www.viadonau.org/fileadmin/user_upload/Manual_on_Danube_Navigation.pdf](https://www.viadonau.org/fileadmin/user_upload/Manual_on_Danube_Navigation.pdf) (accessed: 13.07.2021)
