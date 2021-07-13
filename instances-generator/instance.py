"""
Define instance data structure and instance generator functions.
"""

import json
import os
from typing import Any, Dict, List, Union, IO


class Commodity:
    """A commodity in the network. 
    
    Parameters
    ----------
    id : int
        Unique identifier
    properties : Dict[str, Any]
        Dictionary of property extends of commodity. 
        Note: Commodity volume is a multiple of the instance specific base volume
    type : str
        Commodity type
    group : int
        Group (company) commodity is associated to.
    """

    def __init__(self, id: int, properties: Dict[str, Any], type: str, 
                 lifetime: int = -1, group: int = -1) -> None:
        self.id = id
        self.properties = properties
        self.type = type
        self.lifetime = lifetime
        self.group = group

    def __repr__(self):
        V = self.properties["V"]
        W = self.properties["W"]
        repr_str = f"Commodity {self.id}, V: {V}, W: {W}, type: {self.type}" +\
                   f" dt: {self.lifetime}, group: {self.group} \n"
        return repr_str


class TariffLevel:
    """A discount level in a tariff. 
    
    Parameters
    ----------
    cost_fixed : float
        Fixed cost for each container
    cost_variable : float
        Variable costs for flow
    cost_base : float
        Base costs for choosing this tariff independent of number of containers
        or flow.
    start_y : int
        Number of containers at which this tariff level is applicable.
    """

    def __init__(self, cost_fixed: float, cost_variable: float,
                 cost_base: float, start_y: int) -> None:
        self.cost_fixed = cost_fixed
        self.cost_variable = cost_variable
        self.cost_base = cost_base
        self.start_y = start_y


class Tariff:
    """A tariff applicable on an arc in the time-expanded network. 
    
    Parameters
    ----------
    id : int
        Unique identifier
    levels : List[TariffLevel]
        List of tariff pricing levels. Order of list defines discount levels.
    property_type : str
        Commodity property priced.
    """

    def __init__(self, id: int, levels: List[TariffLevel], property_type: str) -> None:
        self.id = id
        self.levels = levels
        self.property_type = property_type


class NodeBase:
    """A node in the base-network. 
    
    Parameters
    ----------
    id : int
        Unique identifier
    type : str
        Node represents either "source", "facility", "demand" or "bin".
    capacity_storage: Dict[str, int], optional
        Storage capacity for each commodity type. Solver does not use this
        property. Can be used to generate capacitated storage
        arcs in time-expanded network.
    capacity_handling: Dict[str, Dict[str, int]], optional
        Handling capacities for each mode. First keys "inc","out", "tot".
        Inner keys transport modes. Assume capacities const over time.
    facility_type: str
        Facilites can be "warehouse", "crossdock", "trainstation", "port".
        Has no effect for non-facilities.
    region : str
        Region of a facility/demand node. Optional, setting it enables advanced 
        Matheuristic fixing/unfixing strategies.
    data : int
        Optional data to store about node, has no effect on optimization.
        Example: Random instance generation uses this field to store an
                 identifier for external evaluation.
    """
   
    def __init__(self, id: int, type: str, 
                 capacity_storage: Dict[str, int] = None,
                 capacity_handling: Dict[str, Dict[str, int]] = None,
                 facility_type: str = "",
                 region: str = "", 
                 data: int = -1) -> None:
        self.id = id
        self.type = type
        if capacity_storage is None:
            capacity_storage = {}
        self.capacity_storage = capacity_storage
        if capacity_handling is None:
            capacity_handling = {}
        self.capacity_handling = capacity_handling
        self.facility_type = facility_type
        self.region = region
        self.data = data


class Node(NodeBase):
    """Node in the time-expanded network.

    This class provides different methods to add time-dependent stock. 
    Stock can be supplied by a source if a positive integer or represent demand 
    if negative. If storage or handling capacities are not given, they are taken
    from the node base object "node". Non changing stock and capacities for
    each time-period can be given by a list with a single element.

    Parameters
    ----------
    node : NodeBase
        Corresponding node in the base network.
    stock : Dict[int, List[int]], optional
        Stock of Commodity (given by Commodity id) over time. Default: empty dict
    capacity_storage: Dict[str, List[int]], optional
        Storage capacity for each commodity type at each time period. 
        Equals capacity of storage arc in time-expanded network (redundancy).
    #costs_handling : Dict[str, List[Dict[str, float]], optional
        Handling costs for each time period. Key is transporation mode. Second dict has keys
        "inc_y", "inc_f", "out_y", "out_f". 

    Notes
    -----
    A missing Commodity in the stocks-dictionary means neither source
    nor demand property of this node w.r.t. the given Commodity.
    """

    def __init__(self, node: NodeBase, stocks: Dict[int, List[int]] = None,
                 capacity_storage: Dict[str, List[int]] = None) -> None:
        super().__init__(**node.__dict__)
        if stocks is None:
            stocks = {}
        self.stocks = stocks
        if capacity_storage is not None:
            self.capacity_storage = capacity_storage
        else:
            # Convert single capacity value to list
            self.capacity_storage = {k: [v] for k, v in self.capacity_storage.items()}

    def add_stock(self, p_id: int, stock: Union[int, List[int]]) -> None:
        """Add source/demand for a Commodity to node.

        If no stock for a given Commodity is registered, creates stock-entry 
        (and if necessary stock attribute). Otherwise, adds stock to existing
        registered stock.
        
        Parameters
        ----------
        p_id : int
            Commodity ID
        stock : int or List[int]
            Sign-dependent source or demand for p_id. If int, stock is appended,
            if a list, extends stock-entry.
        """
        if not isinstance(stock, list):
            stock = [stock]
        self._add_stock(p_id, stock)

    def _add_stock(self, p_id: int, stock: List[int]) -> None:
        if p_id in self.stocks:
            for t in range(len(stock)):
                self.stocks[p_id][t] += stock[t]
        else:
            self.stocks[p_id] = stock

    def set_stocks(self, stocks: Dict[int, List[int]]) -> None:
        """Set stocks of node and adjust information-state of node-obj. 
        
        Parameters
        ----------
        stocks : Dict[int, List[int]]
            For each Commodity (Commodity-id) define time-dependent source or sink
            property.
        """
        self.stocks = stocks


class ArcBase:
    """A directed arc in the base-network. 
    
    Parameters
    ----------
    id : int
        Unique identifier
    node_orig : Union[int, NodeBase, Node]
        Node the arc originates from. Class instance only stores node-id.
    node_dest : Union[int, NodeBase, Node]
        Node the arc points to. Class instance only stores node-id.
    mode : str
        Transport/Handling/Storage mode
    distance : int
        Distance, e.g. in km
    time : int
        Time it takes to travers arc, e.g. in days
    tariffs: Union[int, Tariff, List[Union[int, Tariff]]]
        Tariff(-ids) available on this arc
    container_capacity : Dict[str, Dict[str, int]]
        Define maximal container capacity for each property (inner dict) 
        and for each commodity type (outer dict). 
    capacity : Dict[str, int]
        Max number of containers for each product type. Value of -1 indicates
        uncapacitated.
    emissions_ykm : List[int]
        For each c_type: Emissions per transport unit and km.
    emissions_fkm : List[int]
        For each c_type: Emissions per unit flow and km. Transport mode => 
        tonne.km, Storage mode => volume.km (with a distance of 1 => volume) 
    emissions_yh : int
        Emissions from handling one transport unit.
    emissions_fh : int
        Emissions from handling one unit of flow. Transport modes => per ton
    cost_yh: float
        Costs for handling one transport unit.
    cost_fh: float
        Costs for handling one transport unit. 
    emissions_y: List[int]
        For each c_type: Total emissions per transport unit. If not set 
        calculated from emissions_ykm[c_type] * distance.
    emissions_f: List[int]
        For each c_type: Total emissions per unit of flow. If not set 
        calculated from emissions_fkm[c_type] * distance.
    """
    
    def __init__(self, id: int, node_orig: Union[int, NodeBase, Node], 
                 node_dest: Union[int, NodeBase, Node], mode: str, 
                 distance: int, time: int, 
                 tariffs: Union[int, Tariff, List[Union[int, Tariff]]], 
                 container_capacity: Dict[str, Dict[str, int]],
                 capacity: Dict[str, int],
                 emissions_ykm: List[int] = [0], emissions_fkm: List[int] = [0],
                 emissions_yh: int = 0, emissions_fh: int = 0,
                 cost_yh: float = 0, cost_fh: float = 0,
                 emissions_y: List[int] = None, emissions_f: List[int] = None) \
                    -> None:
        self.id = id
        self.node_orig = self._get_id(node_orig)
        self.node_dest = self._get_id(node_dest)
        self.mode = mode
        self.distance = distance
        self.time = time
        # Emissions per transport unit
        if emissions_y is None:
            self.emissions_y = []
            for el in emissions_ykm:
                self.emissions_y.append(el * self.distance + emissions_yh)
        else:
            self.emissions_y = emissions_y
        # Emissions per additional unit of flow
        if emissions_f is None:
            self.emissions_f = []
            for el in emissions_fkm:
                self.emissions_f.append(el * self.distance + emissions_fh)
        else:
            self.emissions_f = emissions_f
        self.cost_yh = cost_yh
        self.cost_fh = cost_fh
        if not isinstance(tariffs, list):
            self.tariffs = [self._get_id(tariffs)]
        else:
            self.tariffs = [self._get_id(el) for el in tariffs]
        self.container_capacity = container_capacity
        self.capacity = capacity
        
    def _get_id(self, obj: Union[int, NodeBase, Node, Tariff]) -> int:
        if hasattr(obj, "id"):
            return obj.id
        else:
            return obj


class Arc(ArcBase):
    """An arc in the time-expanded network. 
 
    Parameters
    ----------
    id : int
        Unique identifier for arcs in time-expanded network.
    arc : ArcBase
        Corresponding arc in base-network 
    t_start : int, optional
        Start time period of arc.
    t_end : int, optional
        End time period of arc.
    tariffs: List[Union[int, Tariff]
        Overwrite tariffs available on this arc
    container_capacity : Dict[str, int], optional
        Overwrite maximal transport volume of transporut unit of base arc.
    capacity : Dict[str, int], optional
        Overwrite capacity of base arc.
    """
    
    def __init__(self, id: int, arc: ArcBase, t_start: int = None, t_end: int = None, 
                 tariffs: List[Union[int, Tariff]] = None, 
                 container_capacity: Dict[str, Dict[str, int]] = None,
                 capacity: Dict[str, int] = None) -> None:
        super().__init__(**arc.__dict__)
        self.id = id
        self.t_start = t_start
        self.t_end = t_end
        if tariffs is not None:
            self.tariffs = tariffs
        if container_capacity is not None:
            self.container_capacity = container_capacity
        if capacity is not None:
            self.capacity = capacity


class Instance:
    """Instance for the generalized tactical transport planning 
    problem (GTTP). """

    def __init__(self, instance_properties: List[str] = [], 
                 other: Dict[str, Any] = {}) -> None:
        self.instance_properties = instance_properties
        self.other = other
        self.commodities = []
        self.tariffs = []
        self.nodes = []
        self.arcs = []

    def to_disk(self, fp: str, indent: int, only_json: bool = False) -> None:
        """Write instance as JSON and separate arc file to disc. 
        
        Parameters
        ----------
        fp : str
            File path
        indent : int
            Indent to format JSON-file.
        only_json : bool
            Directly store arcs in JSON. Can result in very large file!
            Default False. 
        """
        if only_json:
            self._dump(fp, indent)
        else:
            self._to_disk(fp, indent)

    def _to_disk(self, fp: str, indent: int) -> None:
        path, file = os.path.split(fp)
        file_name, _ = os.path.splitext(file)
        
        # Construct arcs file path
        arcs_tmp = self.arcs
        fp_arcs = str(os.path.join(path, file_name + ".arcs"))
        self.arcs = file_name + ".arcs"
        
        # Write json
        self._dump(fp, indent)

        # Write .arcs
        self.arcs = arcs_tmp
        self._write_arcs(fp_arcs)

    def _dump(self, fp: str, indent: int) -> None:
        with open(fp, "w") as f:
            json.dump(self, f, default=lambda o: o.__dict__, sort_keys=True,
                      indent=indent)

    def _write_arcs(self, fp: str) -> None:
        with open(fp, "w") as f:
            for arc in self.arcs:
                line = str(arc.id)
                for el in [arc.node_orig, arc.node_dest, arc.t_start, 
                           arc.t_end, arc.mode, arc.distance]:
                    line += " " + str(el)
                for el in zip(arc.emissions_y, arc.emissions_f):
                    line += " " + str(el[0]) + " " + str(el[1])
                for el in [arc.cost_yh, arc.cost_fh]:
                    line += " " + str(el)
                # Write capacities in order defined in c_types and c_properties list
                for c_type in self.other["c_types"]:
                    line += " " + str(arc.capacity[c_type])
                    for c_property in self.other["c_properties"]:
                        line += " " + str(arc.container_capacity[c_type][c_property])
                for tariff_id in arc.tariffs:
                    line += " " + str(tariff_id)
                f.write(line + "\n")
