# Introduction

Instance files are named similarly as described the IPIC publication (see below for the naming convention). However, each instance consists of two files, one ending in .arcs which contains row-wise information on arcs in the network design instance and a .json storing all other information (e.g. commodities, nodes).

An instances consists of a list of tariff, commodity, node and arc objects with the relevant class definitions in [../instance-generator/instance.py](../instance-generator/instance.py).

## .json

Stores the json-serialized tariff, commoditity and node objects of an instance. For a documentation of the individual entries in the .json, please see the class documentations in [../instance-generator/instance.py](../instance-generator/instance.py). 

It also stores an instances specific *other* dictionary, which has the following important elements:
* *time_periods*: Time horizont of instance
* *n_arcs*: Arcs in time-expanded instance
* *n_nodes_base*: Nodes in base network, multiply with time_periods to get nodes in time-expanded network
* *c_types*: List of strings of length 1 of commodity types in instance, its order defines the order of commodity type depending entries in *.arcs* file
* *c_types_n*: Length of c_types
* *c_properties*: List of strings of length 1 of properties of each commodity
* *c_properties_n*: Length of c_properties
* *co2_costs_per_g*: CO2 costs per gram of CO2e emitted in €
* *transport_modes*: List of strings of length 1 of what transport modes are included in this instances (lorry: L, rail: R, ship: S)
* *weight_cost*: Weight given to transport costs objective
* *weight_green*: Weight given to emission costs objective

## .arcs

Stores row-wise the arc information. It follows the following format (the individual entries are documented in the class documention in [../instance-generator/instance.py](../instance-generator/instance.py)):

First 7 columns:

ID node_orig node_dest t_start t_end mode distance 

Next columns repeat for each commodity type in the instance:
(emissions per transport unit (TU)) (emissions per unit of flow) 

Followed by:
(costs of handling one TU) (costs of handling one unit of flow)

Again followed by the following list for each commodity type:
(Capacity in number of TUs installable on this arc) (Volume-capacity of one TU) (Weight-capacity of one TU)

Now, this list is again followed by the following entry for each commodity type:
(ID of Tariff applied on this arc for a given commodity type)

## Naming convention

Instances follow naming scheme r<#>_<#>_<#>_W<#>_C<#>_K<#>_F<#>_T<#>_L/LRS_L/T:

* r<#>_<#>_<#>: r<Number of large regions included>_<Number of medium regions included>_<Number of small regions included>
* W<#>: Number of warehouses included for each of the largest regions
* C<#>: Number of groups (companies) warehouses are grouped into
* K<#>: Total number of commodities
* F<#>: Number of perishable commodities 
* T<#>: Time horizont
* L/LRS: Included transportation modes, either only lorry (L) or lorry, rail (R) and ship (S)
* L/T: If facilities in instance have no handling constraints (loose, L) or have handling constraints (tight, T)

