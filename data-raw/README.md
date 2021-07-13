# Introduction

This folder contains the locations as well as distance and time matrices used for this work. The data is partly based upon test instances released [here](https://plis.univie.ac.at/research/a-matheuristic-for-a-multimodal-long-haul-routing-problem/) by Wolfinger et al. [1]. The following sections describe the data files and how they differ from Wolfinger et al. [1] and how they got extended. Depending on what data you are interested in, please consider to cite Wolfinger et al. [1].

[1] David Wolfinger, Fabien Tricoire, and Karl F. Doerner. *A matheuris-tic for a multimodal long haul routing problem.* In:EURO Journal on Transportation and Logistics8.4(2019), pp.397–433.

## locations.txt

This file follows the following format:

ID;ID';Type;Latitude;Longitude;Name;Size

There are three types of locations differentiated:
1. Random addresses in city regions
2. Ports
3. Train Stations

ID is a unique identifyer for the file. Wolfinger et al. [1] gave only one location for each city region, for this work, we have generated nine additional random addresses for each city region. Furthermore, we have added a size column indicating if the industrial city has a population P of < 100.000 (S), 100.000 <= P < 1.000.000 (M) or P >= 1.000.000 (L).
