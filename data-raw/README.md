# Introduction

This folder contains the locations as well as distance and time matrices used for this work. The data is partly based upon test instances released [here](https://plis.univie.ac.at/research/a-matheuristic-for-a-multimodal-long-haul-routing-problem/) by Wolfinger et al. [1]. The following sections describe the data files and how they differ from Wolfinger et al. [1] and how they got extended. Depending on what data you are interested in, please consider to cite Wolfinger et al. [1].

[1] David Wolfinger, Fabien Tricoire, and Karl F. Doerner. *A matheuristic for a multimodal long haul routing problem.* In:EURO Journal on Transportation and Logistics Volume 8, Issue 4, 2019, pp.397–433.

## locations.txt

This file follows the following format:

ID;ID';Type;Latitude;Longitude;Name;Size

There are three types of locations differentiated:
1. Random addresses in city regions
2. Ports
3. Train Stations

ID is a unique identifyer for each location in the file. Wolfinger et al. [1] gave only one location for each city region, so for this work, we have generated nine additional random addresses for each city region. ID' refers to the original ID used by Wolfinger et al. [1]. Furthermore, we have added a size column indicating if the industrial city has a population P of < 100.000 (S), 100.000 <= P < 1.000.000 (M) or P >= 1.000.000 (L).

## distances.txt

The first two header rows indicate, which rows correspond to which transport mode. It reads as follows: The first 376.382 data rows correspond to road distances for lorries, the next 72 data rows to distances for ships and the last 2970 data rows to rail distances.

A data row has the following format:
ID1;ID2;Distance [km]

ID1 and ID2 denote the IDs of the connected locations from locations.txt. A connection is assumed to be bidirectional.
