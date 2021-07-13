# Introduction

Instance files are named as described the IPIC publication. However, each instance consists of two files, one ending in .arcs which contains row-wise information on arcs in the network design instance and a .json storing all other information (e.g. commodities, nodes). 

## .json

All classes in [../instance-generator/instance.py](../instance-generator/instance.py) are directly serialized into the .json file accept 
See [../instance-generator/instance.py](../instance-generator/instance.py) for a documentation of the individual json-entries. 
