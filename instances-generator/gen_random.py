import copy
from enum import Enum
import math
import random
from typing import Tuple
import statistics

import numpy as np
from scipy.stats import levy
from scipy.stats import norm

from instance import *


# 1 Volume Base Unit (vbu) = 1 m^3
vbu_factor = 1
# 1 Weight Base Unit (wbu) = 1 t
wbu_factor = 1
# Volume & Weight precision in terms of digits right of zero if given in base units
dim_precision = 2
weight_precision = 3
# 40-feet container has 67.5 m^3 volume and 26.48t nutzlast
container_volume = 67.5 * vbu_factor
container_weight = 26.48 * wbu_factor
# Ugly global transport modes list
mode_l = ["L", "R", "S"]


class Size(Enum):
    S = 1
    M = 2
    L = 3


class Region:
    """ A demand region.
    
    Parameters
    ----------
    idx : int
        Demand-Vertex Idx
    name : str
        Region name. Used to index region map
    size : Size
        Region size.
    """
    def __init__(self, idx: int, name: str, size: Size):
        self.idx = idx
        self.name = name
        self.size = size
        self.warehouses = []

    def __repr__(self):
        repr_str = str(self.idx) + " " + self.name + " " + str(self.size) \
                                                + " " + str(self.warehouses)
        if hasattr(self, "port"):
            repr_str += " P:" + str(self.port)
        if hasattr(self, "crossdoc"):
            repr_str += " C:" + str(self.crossdoc)
        if hasattr(self, "train"):
            repr_str += " T:" + str(self.train)
        return repr_str


class VertexType(Enum):
    warehouse = 1
    crossdoc = 2
    port = 3
    train = 4
    demand = 5


class Vertex:
    """ A location. 
     
    Template of a proto-base node, i.e. a soon to become base node of the 
    network.
    
    Parameters
    ----------
    idx : int
        Id in locations file.
    type : VertexType
        Is location a warehouse, crossdoc, port, train or demand node?
    region : str
        Associated region.
    """
    def __init__(self, idx: int, type: VertexType = None, region: str = None):
        self.idx = idx
        self.region = region
        self.type = type
        self.capacity_storage = {}
        self.capacity_handling = {}
    
    def __repr__(self):
        return str(self.idx) + " " + str(self.region) + " " + str(self.type) + "\n"


def _trim_regions_by_size(r_m: Dict[str, Region], v_m: Dict[int, Vertex],
                          W: List[int]) -> None:
    """ Remove unused locations from regions and vertex dictionary. 

    Regions are trimmed to warehouses depending on size property defined in W. 
    Furthermore designates a crossdoc for large regions and defined the demand
    vertex for a region which defines the regions index. Vertex map is updated
    to only include used vertices in the regions dict.
    """
    for key, region in r_m.items():
        if region.size == Size.S:
            idx_l = random.sample(range(0, len(region.warehouses)), k = W[2]+1)
            for c, idx in enumerate(region.warehouses):
                if c not in idx_l:
                    del v_m[idx]
            region.idx = region.warehouses[idx_l[-1]]
            v_m[region.idx].type = VertexType.demand
            w_l = []
            for i, w_idx in enumerate(idx_l):
                if w_idx != idx_l[-1]:
                    w_l.append(region.warehouses[w_idx])
            region.warehouses = w_l

        if region.size == Size.M:
            idx_l = random.sample(range(0, len(region.warehouses)), W[1]+1)
            for c, idx in enumerate(region.warehouses):
                if c not in idx_l:
                    del v_m[idx]
            region.idx = region.warehouses[idx_l[-1]]
            v_m[region.idx].type = VertexType.demand
            w_l = []
            for i, w_idx in enumerate(idx_l):
                if w_idx != idx_l[-1]:
                    w_l.append(region.warehouses[w_idx])
            region.warehouses = w_l

        if region.size == Size.L:
            idx_l = random.sample(range(0, len(region.warehouses)), W[0]+2)
            for c, idx in enumerate(region.warehouses):
                if c not in idx_l:
                    del v_m[idx]
            region.crossdoc = region.warehouses[idx_l[-2]]
            v_m[region.crossdoc].type = VertexType.crossdoc
            region.idx = region.warehouses[idx_l[-1]]
            v_m[region.idx].type = VertexType.demand
            w_l = []
            for i, w_idx in enumerate(idx_l):
                if i < len(idx_l) - 2:
                    w_l.append(region.warehouses[w_idx])
            region.warehouses = w_l


def read_locations(locations_f: str, W: List[int]) -> None:
    """ Reads in locations from file locations_f.

    Fills and returns a regions and vertex map. Both are trimmed to 
    adapt available locations to warehouses per region sizes specified
    in W.
    """
    region_idx = 0
    r_m = {}
    v_m = {}
    with open(locations_f, "r") as f:
        for line in f:
            l = line.split(";")
            idx = int(l[0])
            # Case Warehouse
            if int(l[2]) == 1:
                v_type = VertexType.warehouse
                region = l[5]
                l[6] = l[6][:-1]
                if l[6] == 'S':
                    size = Size.S
                elif l[6] == 'M':
                    size = Size.M
                else:
                    size = Size.L
                if region not in r_m:
                    r_m[region] = Region(region_idx, region, size)
                    region_idx += 1
                v = Vertex(idx, v_type, region)
                r_m[region].warehouses.append(v.idx)
            # Read Ports
            if int(l[2]) == 2:
                v_type = VertexType.port
                region = l[5][5:] # cut prefix "Port "
                v = Vertex(idx, v_type, region)
                r_m[region].port = v.idx
            # Read Train Stations
            if int(l[2]) == 3: 
                v_type = VertexType.train
                region = l[5][13:] # cut prefix "Trainstation "
                v = Vertex(idx, v_type, region)
                r_m[region].train = v.idx
            v_m[v.idx] = v
    # Post Process Regions to designate demand node and cross-dock & trim by size
    _trim_regions_by_size(r_m, v_m, W)
    
    return r_m, v_m


def get_max_vertex_id(v_m: Dict[int, Vertex]) -> int:
    """ Return maximum vertex id in v_m. """
    max_id = 0
    for key, vertex in v_m.items():
        if vertex.idx > max_id:
            max_id = vertex.idx
    return max_id


def select_regions(r_m: Dict[str, Region], v_m: Dict[int, Vertex], 
                   R: List[int], use_regions: List[str]) -> None:
    """ Select user defined regions from all regions r_l.
    
    Select either those regions specified in use_regions by name or randomly
    select a given size-split defined by R. Selection overwrides r_m.
    Updates vertex map v_m to only include chosen vertices, does not delete
    transshipment nodes!
    """
    if use_regions is not None:
        _select_regions_by_name(r_m, v_m, use_regions)
    else:
        _select_regions_by_size_split(r_m, v_m, R)


def _select_regions_by_name(r_m, v_m, use_regions):
    print(use_regions)
    # Remove not selected regions
    for key in list(r_m):
        if key not in use_regions:
            del r_m[key]
    # Update surviving vertices
    for key in list(v_m):
        ts_types = [VertexType.crossdoc, VertexType.port, VertexType.train]
        if v_m[key].type not in ts_types:
            if v_m[key].region not in use_regions:
                del v_m[key]


def _split_regions_by_size(r_m) -> Tuple[List[str], List[str], List[str]]:
    """ Returns 3 lists containing in that order S, M and L region names."""
    s_names = []
    m_names = []
    l_names = []
    for key, region in r_m.items():
        if region.size == Size.S:
            s_names.append(region.name)
        elif region.size == Size.M:
            m_names.append(region.name)
        else:
            l_names.append(region.name)
    return s_names, m_names, l_names


def _select_regions_by_size_split(r_m, v_m, R):
    # count large, medium and small regions
    s_names, m_names, l_names = _split_regions_by_size(r_m)
    # generate chosen index sets
    s_idx_l = random.sample(range(0, len(s_names)), k = R[2])
    m_idx_l = random.sample(range(0, len(m_names)), k = R[1])
    l_idx_l = random.sample(range(0, len(l_names)), k = R[0])
    idx_ll = [s_idx_l, m_idx_l, l_idx_l]
    names_ll = [s_names, m_names, l_names]
    # Select only chosen regions from region map
    r_new = {}
    for idx_l, names_l in zip(idx_ll, names_ll):
        for idx in idx_l:
            name = names_l[idx]
            r_new[name] = r_m[name]
    # Update original regions map
    for key in list(r_m):
        if key not in r_new:
            del r_m[key]
    # Update surviving vertices
    ts_types = [VertexType.crossdoc, VertexType.port, VertexType.train]
    for key in list(v_m):
        if v_m[key].type not in ts_types:
            if v_m[key].region not in r_m:
                del v_m[key]


def read_distance_matrix(distance_matrix_f: str, v_m: Dict[int, Vertex]) \
        -> Tuple[Dict[int, Dict[int, int]], Dict[int, Dict[int, int]], Dict[int, Dict[int, int]]]:
    """ Read and return distances between vertices in instance by mode. 
    
    Returns distance-dictionaries in that order: lorry, ship, train.
    """
    d_l = {}
    d_s = {}
    d_t = {}
    with open(distance_matrix_f, "r") as f:
        # Read modes
        line = f.readline()
        l = line.split(" ")
        if len(l) != 3:
            raise NotImplementedError("Distance Matrix has " + len(l) + 
                                      " modes, expected 3")
        line = f.readline()
        l = line.split(" ")
        l_count = int(l[0])
        s_count = int(l[1])
        t_count = int(l[2])
        # Parse lorry distances
        for i in range(0, l_count):
            line = f.readline()
            l = line.split(" ")
            v_o = int(l[0])
            v_d = int(l[1])
            d = int(l[2])
            if v_o not in v_m or v_d not in v_m:
                continue
            if v_o not in d_l:
                d_l[v_o] = {}
            d_l[v_o][v_d] = d
        # Parse ship distances
        for i in range(0, s_count):
            line = f.readline()
            l = line.split(" ")
            v_o = int(l[0])
            v_d = int(l[1])
            d = int(l[2])
            if v_o not in d_s:
                d_s[v_o] = {}
            d_s[v_o][v_d] = d
        # Parse train distances
        for i in range(0, t_count):
            line = f.readline()
            l = line.split(" ")
            v_o = int(l[0])
            v_d = int(l[1])
            d = int(l[2])
            if v_o not in d_t:
                d_t[v_o] = {}
            d_t[v_o][v_d] = d
    return d_l, d_s, d_t


def _get_closet_idx(origin: int, dest_l: List[int], 
                   d: Dict[int, Dict[int, int]]) -> int:
    """ Return closest point in dest_l to origin using distance matrix d."""
    min_d = -1
    min_idx = -1
    for i, destination in enumerate(dest_l):
        if i == 0:
            min_d = d[origin][destination]
            min_idx = destination
        else:
            dist = d[origin][destination]
            if dist < min_d:
                min_d = dist
                min_idx = destination
    return min_idx


def _build_transshipment_points_list(v_m) \
                                    -> Tuple[List[int], List[int], List[int]]:
    """ Returns index sets for crossdocs, ports and trainstation in that order."""
    ts_crossdoc_l = []
    ts_port_l = []
    ts_train_l = []
    for v in v_m.values():
        if v.type == VertexType.crossdoc:
            ts_crossdoc_l.append(v.idx)
        if v.type == VertexType.port:
            ts_port_l.append(v.idx)
        if v.type == VertexType.train:
            ts_train_l.append(v.idx)
    return ts_crossdoc_l, ts_port_l, ts_train_l


def _remove_idx_from_matrix(idx: int, m: Dict[int, Dict[int, int]]) -> None:
    if idx in m:
        del m[idx]
    for key in list(m):
        if idx in m[key]:
            del m[key][idx]


def select_closest_transshipment(r_m, v_m, d_l, d_s, d_t, direct_delivery) \
                                                                    -> None:
    """ Removes from vertex list transshipment points not closet to any region."""
    # Build transshipment points list
    ts_crossdoc_l, ts_port_l, ts_train_l = _build_transshipment_points_list(v_m)
    # Get closet transshipment points, direct-delivery instances have none
    closest_s = set()
    if not direct_delivery:
        for region in r_m.values():
            closest_s.add( _get_closet_idx(region.idx, ts_crossdoc_l, d_l) )
            closest_s.add( _get_closet_idx(region.idx, ts_port_l, d_l) )
            closest_s.add( _get_closet_idx(region.idx, ts_train_l, d_l) )
    # Remove transshipment points not closet to any region
    ts_types = [VertexType.crossdoc, VertexType.port, VertexType.train]
    for v_idx in list(v_m):
        if v_m[v_idx].type in ts_types:
            if v_idx not in closest_s:
                del v_m[v_idx]
                # Update distance matrices
                for d in [d_l, d_s, d_t]:
                    _remove_idx_from_matrix(v_idx, d)
            

def read_traveltime_matrix(time_matrix_f: str, v_m: Dict[int, Vertex]) \
        -> Tuple[Dict[int, Dict[int, int]], Dict[int, Dict[int, int]], Dict[int, Dict[int, int]]]:
    """ Read and return time matrices between vertices in instance by mode. 
    
    Returns traveltime-dictionaries in that order: lorry, ship, train.
    """
    t_l = {}
    t_s = {}
    t_t = {}
    with open(time_matrix_f, "r") as f:
        # Read modes
        line = f.readline()
        l = line.split(" ")
        if len(l) != 3:
            raise NotImplementedError("Traveltime Matrix has " + len(l) + 
                                      " modes, expected 3")
        line = f.readline()
        l = line.split(" ")
        l_count = int(l[0])
        s_count = int(l[1])
        t_count = int(l[2])
        # Parse lorry time
        for i in range(0, l_count):
            line = f.readline()
            l = line.split(" ")
            v_o = int(l[0])
            v_d = int(l[1])
            t = int(l[2])
            if v_o not in v_m or v_d not in v_m:
                continue
            if v_o not in t_l:
                t_l[v_o] = {}
            t_l[v_o][v_d] = t
        # Parse ship time
        for i in range(0, s_count):
            line = f.readline()
            l = line.split(" ")
            v_o = int(l[0])
            v_d = int(l[1])
            t = int(l[2])
            if v_o not in v_m or v_d not in v_m:
                continue
            if v_o not in t_s:
                t_s[v_o] = {}
            t_s[v_o][v_d] = t
        # Parse train time
        for i in range(0, t_count):
            line = f.readline()
            l = line.split(" ")
            v_o = int(l[0])
            v_d = int(l[1])
            t = int(l[2])
            if v_o not in v_m or v_d not in v_m:
                continue
            if v_o not in t_t:
                t_t[v_o] = {}
            t_t[v_o][v_d] = t
    return t_l, t_s, t_t


def _get_v_of_region(region, v_m):
    """ Return list of vertex-ids in region region. """
    v_l = []
    for v_id, v in v_m.items():
        if v.region == region:
            v_l.append(v_id)
    return v_l


def correct_time_matrix_l(t_matrix_l: Dict[int, Dict[int, int]], 
                          v_m: Dict[int, Vertex],
                          r_m: Dict[str, Region]) -> None:
    """ Correct s.t. lorries from one region to another always need same days.
    
    Correct t_matrix_l s.t. a lorry from region A always takes the same number
    of days to region B independent of location in region A or B. This prevents
    one element in A having e.g. only 1 day travel time and all others 2 days
    => unrealistic!
    """
    for region in r_m:
        v_l = _get_v_of_region(region, v_m)
        for other_region in r_m:
            if other_region == region:
                continue
            v_l_other = _get_v_of_region(other_region, v_m)
            t_l = []
            for v1 in v_l:
                for v2 in v_l_other:
                    t_l.append(t_matrix_l[v1][v2])
            t_median = statistics.median_high(t_l)
            if t_median != statistics.mean(t_l):
                print( "Changing regions " + region +"->" + other_region +\
                       " to t:" + str(t_median) )
                for v1 in v_l:
                    for v2 in v_l_other:
                        t_matrix_l[v1][v2] = t_median
            

def _is_in_unique_region(w, w_l, v_m):
    """ Return True if warehouse w shares no region with warehouses in w_l. """
    w_region = v_m[w].region
    for w2 in w_l:
        w2_region = v_m[w2].region
        if w2_region == w_region:
            return False
    return True


def associate_warehouses_to_group(group_w_l: List[List[int]], 
                            v_m: Dict[int, Vertex], split: int = 3) -> None:
    """ Associate warehouses in v_m to group in group_w_l.

    split ... 2/split groups have 1 warehouse, 1/split have 2, ...
    """
    # Create Warehouses List
    w_l = []
    for key, vertex in v_m.items():
        if vertex.type == VertexType.warehouse:
            w_l.append(key)
 
    comp_idx_l = [i for i in range(0, len(group_w_l))]
    depth = 0
    while len(w_l) > 0:
        if depth == 0:
            # Associate a warehouse to each company
            w_idx_l = random.sample(range(0, len(w_l)), k=len(group_w_l))
            for c_idx, w_idx in enumerate(w_idx_l):
                w = w_l[w_idx]
                group_w_l[c_idx].append(w)
        else:
            k = min(math.ceil(len(comp_idx_l) / 3), len(w_l))
            w_idx_l = random.sample(range(0, len(w_l)), k=k)
            for w_idx in w_idx_l:
                w = w_l[w_idx]
                associated = False
                # Try to associate warehouse to a company not having a 
                # warehouse in same region 
                for c_idx in comp_idx_l:
                    if len(group_w_l[c_idx]) == depth:
                        if _is_in_unique_region(w, group_w_l[c_idx], v_m):
                            group_w_l[c_idx].append(w)
                            associated = True
                            break
                # Can't help if its not possible
                if not associated:
                    for c_idx in comp_idx_l:
                        if len(group_w_l[c_idx]) == depth:
                            group_w_l[c_idx].append(w)
                            associated = True
                            break
                assert( associated == True )
        # Remove idx not used for association
        for i in range(0, len(group_w_l)):
            if len(group_w_l[i]) == depth:
                comp_idx_l.remove(i)
        # Remove associated warehouses
        w_new = []
        w_new.extend(w_l)
        for w_idx in w_idx_l:
            w = w_l[w_idx]
            w_new.remove(w)
        w_l = w_new
        depth += 1


def init_commodities(K: int, F: int) -> List[Commodity]:
    """ Return list of K commodities of which last F are perishable. 
    
    Id of Commodity coincides with position in list. Property is only used
    in set_stock function if a change is every necessary.
    """
    min_dim = 0.01 # m^3
    max_dim = 2.16 # m^3
    min_dens = 50 # kg / m^3
    max_dens = 1200 # kg / m^3
    # density: 1 t / m^3 = 1 / 100 t / vbu = 100 * 1000/100 wbu/vbu
    # => density: 1 t / m^3 = 1000 wbu / vbu 
    # => denisty: 1 kg / m^3 = 1 wbu / vbu
    # Levy Distribution
    location = 0
    scale = 0.2
    k_l = []
    min_v = 1000
    max_v = 0
    min_w = 1000
    max_w = 0
    for i in range(K):
        # Calculate Property Extent
        dim = max_dim + 1
        while dim > max_dim:
            dim = levy.rvs(loc=location, scale=scale, size=1)
        dim_m3 = round(max(dim[0], min_dim), 2)
        dens = random.randrange(min_dens, max_dens, step=1)
        weight_kg = round(dim_m3*dens, 2)
        weight_t = weight_kg / 1000
        if weight_precision > 0:
            weight_wbu = round(weight_t * wbu_factor, weight_precision)
        else:
            weight_wbu = round(weight_t * wbu_factor)
        if dim_precision > 0:
            dim_vbu = round(dim_m3 * vbu_factor, dim_precision)
        else:
            dim_vbu = round(dim_m3 * vbu_factor)
        # Type
        if i >= K - F:
            c_type = "F" #fresh
        else:
            c_type = "N" #dry
        k_l.append(Commodity(i, {"V": dim_vbu, "W": weight_wbu}, c_type))
        if dim_vbu > max_v:
            max_v = dim_vbu
        if dim_vbu < min_v:
            min_v = dim_vbu
        if weight_wbu > max_w:
            max_w = weight_wbu
        if weight_wbu < min_w:
            min_w = weight_wbu
    return k_l


def associate_products_to_group(group_k_l: List[List[int]], 
                group_w_l: List[List[int]], k_l: List[Commodity], 
                F: int) -> None:
    """ Associate each product to a group, probability based on group size."""
    group_sizes = [len(el) for el in group_w_l]
    n_groups = len(group_sizes)
    n_warehouses = sum(group_sizes)
    p = np.array([el / n_warehouses for el in group_sizes])
    idx_l = [el for el in range(len(group_k_l))]
    # Associate half of normal products evenly to groups
    n_normal = len(k_l) - F
    n_normal_associated = 0
    n_normal_halve = round(n_normal / 2)
    while n_normal_associated < n_normal_halve:
        group_idx_l = np.random.permutation(n_groups)
        for group_idx in group_idx_l:
            group_k_l[group_idx].append(k_l[n_normal_associated].id)
            n_normal_associated += 1
            if n_normal_associated == n_normal_halve:
                break
    # Associated second halve of non-perishable commodities with probability
    # proportional to group sizes
    while n_normal_associated < n_normal:
        group_idx = np.random.choice(idx_l, p = p)
        group_k_l[group_idx].append(k_l[n_normal_associated].id)
        n_normal_associated += 1
    # Associate half of fresh products evenly to groups
    n_fresh_halve = round(F / 2)
    while n_normal_associated < n_normal + n_fresh_halve:
        group_idx_l = np.random.permutation(n_groups)
        for group_idx in group_idx_l:
            group_k_l[group_idx].append(k_l[n_normal_associated].id)
            n_normal_associated += 1
            if n_normal_associated == n_normal + n_fresh_halve:
                break
    # Associated second halve of fresh commodities with probability
    # proportional to group sizes
    while n_normal_associated < len(k_l):
        group_idx = np.random.choice(idx_l, p = p)
        group_k_l[group_idx].append(k_l[n_normal_associated].id)
        n_normal_associated += 1


def associate_products_to_demandregions(k_r_m: Dict[int, List[int]], 
                    group_k_l: List[List[int]], group_w_l: List[List[int]], 
                    v_m: Dict[int, Vertex], r_m: Dict[str, Region]) -> None:
    """ Associate to each commodity its requested demand regions using k_r_m. """
    n_regions = len(r_m)
    regions = list(r_m.keys())
    for group_idx, k_group in enumerate(group_k_l):
        # Extract regions group has subsidiaries in
        w_group = group_w_l[group_idx]
        regions_of_group = []
        for w in w_group:
            region = v_m[w].region
            if region not in regions_of_group:
                regions_of_group.append(region)
        n_regions_of_group = len(regions_of_group)
        available_regions = [region for region in regions if region not in regions_of_group]
        # Associate to each product one or multiple demand regions
        for k_id in k_group:
            # Calculate number of demand regions which demand product
            p = random.randint(25, 100) / 100
            n_demand = round(max((n_regions - n_regions_of_group) * p, 1))
            # Calculate draw-probability of each possible demand region
            p = []
            for region in available_regions:
                if r_m[region].size == Size.S:
                    p.append(1)
                elif r_m[region].size == Size.M:
                    p.append(2)
                else:
                    p.append(4)
            p = np.array(p)
            p = p / np.sum(p)
            # Draw demand regions
            r_idx_a = np.random.choice(len(available_regions), size=n_demand,    
                                       replace=False, p=p)
            for r_idx in r_idx_a:
                if k_id not in k_r_m:
                    k_r_m[k_id] = []
                k_r_m[k_id].append(available_regions[r_idx])


def _get_commodity_types(k_l: List[Commodity]) -> List[str]:
    type_l = [] 
    for k in k_l:
        if k.type not in type_l:
            type_l.append(k.type)
    return type_l


def gen_bin_node(max_vertex_id: int, k_l: List[Commodity]) -> Node:
    """ Return bin node with ID set to max-id in location file + 1. """
    # Set default storage capacity
    k_type_l = _get_commodity_types(k_l)
    capacity_storage = {}
    for k_type in k_type_l:
        capacity_storage[k_type] = 0
    bin_b = NodeBase(max_vertex_id + 1, "bin", capacity_storage)
    return Node(bin_b)


def gen_demand_nodes(r_m: Dict[str, Region], 
                     k_l: List[Commodity]) -> Dict[str, Node]:
    """ Return region (name) to demand node map. """
    # Set default storage capacity
    k_type_l = _get_commodity_types(k_l)
    capacity_storage = {}
    for k_type in k_type_l:
        capacity_storage[k_type] = 0
    # Generate demand nodes
    demand_node_m = {}
    for region in r_m:
        d_idx = r_m[region].idx
        node_b = NodeBase(d_idx, "demand", capacity_storage, region=region)
        demand_node_m[region] = Node(node_b)
    return demand_node_m


def gen_warehouse_nodes(v_m: Dict[int, Vertex], 
                        k_l: List[Commodity]) -> Dict[int, Node]:
    """ Return vertex/location-id to warehouse node map."""
    # Set default storage capacity
    k_type_l = _get_commodity_types(k_l)
    capacity_storage = {}
    for k_type in k_type_l:
        capacity_storage[k_type] = 0
    # Set default handling capacity
    capacity_handling = {}
    for key in {"inc", "out", "tot"}:
        capacity_handling[key] = {}
        for mode in ["L"]:
            capacity_handling[key][mode] = 0
    # Generate warehouse nodes
    warehouse_node_m = {}
    for v_id, v in v_m.items():
        if v.type == VertexType.warehouse:
            node_b = NodeBase(v_id, "facility", capacity_storage, 
                          capacity_handling, "warehouse", v.region)
            warehouse_node_m[v_id] = Node(node_b)
    return warehouse_node_m


def  _get_closest_warehouse(d_node: Node, w_group: List[int], d_l) -> int:
    """ Assume node id is vertex/location id used in distance matrix. """
    min_d = -1
    min_w_idx = -1
    for w_idx in w_group:
        d = d_l[w_idx][d_node.id]
        if d < min_d or min_d == -1:
            min_d = d
            min_w_idx = w_idx
    return min_w_idx


def _draw_average_demand(max_demand: int, k: Commodity, region_size: Size):
    k_w = round(k.properties["W"] * 10**weight_precision)
    max_demand = round(max_demand * wbu_factor * 10**weight_precision)
    demand = random.randint(k_w, max_demand)
    if region_size == Size.S:
        demand = 0.9 * demand
    if region_size == Size.L:
        demand = 1.1 * demand
    demand = math.ceil(demand / k_w)
    return demand


def _set_demand_pattern(d_node: Node, k: Commodity, T:int, cycle_size: int, 
                        min_travel_t: int, v_m: Dict[int, Vertex], 
                        r_m: Dict[str, Region], max_demand: int) -> int:
    """ Create commodity demand in d_node. Return average requested products."""
    # Calculate Stock
    stock = [0 for el in range(T)]
    avg_demand = _draw_average_demand(max_demand, k, 
                                      r_m[v_m[d_node.id].region].size)
    for t in range(min_travel_t, T, cycle_size):
        demand = -1
        while demand < 1:
            demand = round(norm.rvs(loc=avg_demand, scale=0.1*avg_demand))
        stock[t] = -demand
    # Set Stock
    d_node.add_stock(k.id, stock)
    return avg_demand


def _set_warehouse_stock(w_node: Node, d_node: Node, k: Commodity, 
                         min_travel_t: int, avg_demand: int) -> None:
    # Warehouse stock for direct demand fullfillment
    demand_l = d_node.stocks[k.id]
    stock = [0 for el in range(len(demand_l))]
    for t, demand in enumerate(demand_l):
        if t < min_travel_t:
            if demand != 0:
                print("infeasible")
            continue
        # Generate stock only in time periods with non-zero demand
        if demand < 0:
            stock_t = round(avg_demand * 1.2)
            stock[t - min_travel_t] = max(stock_t, -demand)
    w_node.add_stock(k.id, stock)


def set_stock(T: int, d_l: Dict[int, Dict[int, int]], 
              t_matrix_l: Dict[int, Dict[int, int]],
              demand_node_m: Dict[str, Node],
              warehouse_node_m: Dict[int, Node], 
              group_w_l: List[List[int]], group_k_l: List[List[int]], 
              k_r_m: Dict[int, List[str]], k_l: List[Commodity], 
              r_m: Dict[str, Region], v_m: Dict[int, Vertex],
              max_demand: int) -> None:
    """ Set stock of demand and warehouse nodes. 
    
    Return average regional product demand map.
    """
    # Define product demand cycles
    k_cycle_m = {}
    for k in k_l:
        if T > 7:
            k_cycle_m[k.id] = random.choice([1, 2, 7])
        else:
            k_cycle_m[k.id] = random.choice([1, 2])
    # For each commodity, set stocks
    r_demand_m = {}
    for group_idx, group_k in enumerate(group_k_l):
        for k_id in group_k:
            demand_region_l = k_r_m[k_id]
            for demand_region in demand_region_l:
                # Find closest warehouse
                d_node = demand_node_m[demand_region]
                w_group = group_w_l[group_idx]
                closest_w_idx = _get_closest_warehouse(d_node, w_group, d_l)
                # Set demand cycles starting at first reachable time period
                min_travel_t = t_matrix_l[closest_w_idx][d_node.id]
                avg_demand = _set_demand_pattern(d_node, k_l[k_id], T, 
                                                 k_cycle_m[k_id], min_travel_t,  
                                                 v_m, r_m, max_demand)
                # Set warehouse stocks shifted for min_travel_t
                _set_warehouse_stock(warehouse_node_m[closest_w_idx], d_node, 
                                     k_l[k_id], min_travel_t, avg_demand)
                # Create product entry for average regional demand map
                if demand_region not in r_demand_m:
                    r_demand_m[demand_region] = {}
                r_demand_m[demand_region][k_id] = avg_demand
    return r_demand_m


def set_lifetime(K: int, F: int, d_l: Dict[int, Dict[int, int]], 
                 t_l: Dict[int, Dict[int, int]],
                 demand_node_m: Dict[str, Node],
                 group_w_l: List[List[int]], group_k_l: List[List[int]], 
                 k_r_m: Dict[int, List[str]], k_l: List[Commodity]) -> None:
    """Set lifetime of each perishable good.

    Lifetime is set uniformly between the minimum travel time of a truck to 
    the most distant associated demand node max_min_t and max_min_t + 4.
    """
    # Iterate all commodities 
    min_fresh_k_id = K - F
    for group_idx, group_k in enumerate(group_k_l):
        for k_id in group_k:
            # Only relevant for fresh products
            if k_id >= min_fresh_k_id:
                # Find max { minimum travel time to a demand region }
                max_min_t = -1
                demand_region_l = k_r_m[k_id]
                for demand_region in demand_region_l:
                    # Find warehouse servicing demand_region
                    d_node = demand_node_m[demand_region]
                    w_group = group_w_l[group_idx]
                    closest_w_idx = _get_closest_warehouse(d_node, w_group, d_l)
                    min_travel_t = t_l[d_node.id][closest_w_idx]
                    if min_travel_t > max_min_t:
                        max_min_t = min_travel_t
                # Set lifetime
                k_l[k_id].lifetime = random.randint(max_min_t, max_min_t + 4)


def set_commodity_group(group_k_l: List[List[int]], 
                        k_l: List[Commodity]) -> None:
    """ Set group-attribute of each commodity. """
    for group_idx, group_k in enumerate(group_k_l):
        for k_id in group_k: 
            k_l[k_id].group = group_idx  


def set_storage_capacities(T: int, warehouse_node_m: Dict[int, Node], 
                           k_l: List[Commodity], r_m: Dict[str, Region], 
                           v_m: Dict[int, Vertex]) -> None:
    """ Set storage capacities of warehouses. """
    min_inv_level = 0.5
    max_inv_level = 0.8
    k_type_l = _get_commodity_types(k_l)
    # Init variables for average capacity statistics
    avg_s = {}
    n_s = {}
    avg_m = {}
    n_m = {}
    avg_l = {}
    n_l = {}
    for d in [avg_s, n_s, avg_m, n_m, avg_l, n_l]:
        for k_type in k_type_l:
            d[k_type] = 0
    empty_warehouse_ids = {}
    for k_type in k_type_l:
            empty_warehouse_ids[k_type] = []
    # Set capacity of warehouses with stock
    for w_idx, warehouse in warehouse_node_m.items():
        stock_w_sum = {}
        for i, k_type in enumerate(k_type_l):
            # Calculate volume of total generated stock in each time period
            stock_v_sum = [0 for t in range(T)]
            stock_w_sum[k_type] = [0 for t in range(T)] 
            for k_id, stock_l in warehouse.stocks.items():
                # Count only commodities of same commodity type
                if k_l[k_id].type != k_type:
                    continue
                for t in range(T):
                    stock_v_sum[t] += stock_l[t] * k_l[k_id].properties["V"]
                    stock_w_sum[k_type][t] += stock_l[t] * k_l[k_id].properties["W"]
            # Get maximal generated stock in a time period
            max_v_stock = max(stock_v_sum)
            if max_v_stock == 0:
                empty_warehouse_ids[k_type].append(w_idx)
            else:
                # Draw capacity
                min_capacity = round(max_v_stock / max_inv_level)
                max_capacity = round(max_v_stock / min_inv_level)
                capacity = random.randint(min_capacity, max_capacity)
                # Set new capacity
                warehouse.capacity_storage[k_type] = [capacity]
                # Statistics
                r_size = r_m[v_m[w_idx].region].size
                if r_size == Size.S:
                    avg_s[k_type] += capacity
                    n_s[k_type] += 1
                elif r_size == Size.M:
                    avg_m[k_type] += capacity
                    n_m[k_type] += 1
                else:
                    avg_l[k_type] += capacity
                    n_l[k_type] += 1
            # Record maximum weight of stock for later setting of handling cap.
            if i == 0:
                max_w_stock = {}
            max_w_stock[k_type] = max(stock_w_sum[k_type])
        warehouse.max_w_stock = max_w_stock
    # Set capacities of totally empty warehouses
    totally_empty_warehouse_ids = []
    for i, k_type in enumerate(k_type_l):
        # Find warehouses without any stock of any type
        if i == 0:
            totally_empty_warehouse_ids = empty_warehouse_ids[k_type]
        else:
            h_l = list(totally_empty_warehouse_ids)
            for w_id in h_l:
                if w_id not in empty_warehouse_ids[k_type]:
                    totally_empty_warehouse_ids.remove(w_id)
    # Set their capacity levels to average stock for region size
    for w_idx in totally_empty_warehouse_ids:
        warehouse = warehouse_node_m[w_idx]
        region_size = r_m[v_m[w_idx].region].size
        for k_type in k_type_l:
            if region_size == Size.S and n_s[k_type] > 0:
                half = round(avg_s[k_type] / n_s[k_type] / 2)
                warehouse.capacity_storage[k_type] = [half]
            if region_size == Size.M and n_m[k_type] > 0:
                half = round(avg_m[k_type] / n_m[k_type] / 2)
                warehouse.capacity_storage[k_type] = [half]
            if region_size == Size.L and n_l[k_type] > 0:
                half = round(avg_l[k_type] / n_l[k_type] / 2)
                warehouse.capacity_storage[k_type] = [half]


def gen_transshipment_nodes(v_m: Dict[int, Vertex],
                            k_l: List[Commodity]) -> Dict[int, Node]:
    """ Generate transshipment nodes map. 
    
    Nodes are constructed without storage but infinite handling capacity.
    """
    k_type_l = _get_commodity_types(k_l)
    # Create nodes
    transshipment_node_m = {}
    for v_id, v in v_m.items():
        # Set default storage capacity to zero
        capacity_storage = {}
        for k_type in k_type_l:
            capacity_storage[k_type] = 0
        # Set default handling capacity to infinity
        capacity_handling = {}
        for key in {"inc", "out", "tot"}:
            capacity_handling[key] = {}
            capacity_handling[key]["L"] = -1
            if v.type == VertexType.port:
                capacity_handling[key]["S"] = -1
            if v.type == VertexType.train:
                capacity_handling[key]["R"] = -1
        if v.type == VertexType.demand or v.type == VertexType.warehouse:
            continue
        region = f"{v.region}"
        if v.type == VertexType.port:
            facility_type = "port"
            if "S" not in mode_l:
                # Do not create port node
                continue
        if v.type == VertexType.crossdoc:
            facility_type = "crossdock"
        if v.type == VertexType.train:
            facility_type = "trainstation"
            if "R" not in mode_l:
                # Do not create train station
                continue
        node_b = NodeBase(v_id, "facility", capacity_storage, 
                          capacity_handling, facility_type, region)
        transshipment_node_m[v_id] = Node(node_b)
    return transshipment_node_m


def _set_handling_capacities_warehouses(warehouse_node_m: Dict[int, Node]):
    """ CURRENTLY ALWAYS SETS INFINITE HANDLING CAPACITY! """
    max_capacity_ratio = 1
    for w_id, warehouse in warehouse_node_m.items():
        # Calculate maximum serviceable truck numbers
        n_container_v = 0
        n_container_w = 0
        for k_type, capacity_l in warehouse.capacity_storage.items():
            if len(capacity_l) == 1:
                capacity = capacity_l[0]
            else:
                raise NotImplementedError("_set_handling_capacities_warehouses:"
                                          + "Changing capacity over time not"
                                          + "implemented. ")
            n_container_v += math.ceil(capacity / container_volume * max_capacity_ratio)
            n_container_w += math.ceil(warehouse.max_w_stock[k_type] / container_weight)
        del warehouse.max_w_stock
        # Set handling capacities
        capacity_handling = {}
        for key in {"inc", "out", "tot"}:
            capacity_handling[key] = {}
            capacity_handling[key]["L"] = -1#max(n_container_v, n_container_w)
        warehouse.capacity_handling = capacity_handling


def _get_closest_region(t_node: Node, r_m: Dict[int, Region], 
                        v_m: Dict[int, Vertex],
                        d_l: Dict[int, Dict[int, int]]) -> str:
    t_region = v_m[t_node.id].region
    # If transshipment point is in a region with warehouses, problem solved
    if t_region in r_m:
        return t_region
    # Else, look for closest region in r_m
    min_region = ""
    min_d = -1
    for r_name, region in r_m.items():
        d = d_l[t_node.id][region.idx]
        if d < min_d or min_d == -1:
            min_region = r_name
            min_d = d
    return min_region


def _set_handling_capacities_transshipment(transshipment_node_m, 
                                           r_demand_m, k_l, r_m, v_m, d_l):
    demand_fraction = 0.3
    for t_idx, t_node in transshipment_node_m.items():
        closet_region = _get_closest_region(t_node, r_m, v_m, d_l)
        # Sum average demand (weight) for region
        sum_demand_w = 0
        sum_demand_v = 0
        for k_id, avg_demand in r_demand_m[closet_region].items():
            sum_demand_w += avg_demand * k_l[k_id].properties["W"]
            sum_demand_v += avg_demand * k_l[k_id].properties["V"]
        capacity_w = math.ceil(sum_demand_w * demand_fraction / container_weight)
        capacity_v = math.ceil(sum_demand_v * demand_fraction / container_volume)
        capacity = max(capacity_w, capacity_v)
        t_vertex = v_m[t_idx]
        if t_vertex.type == VertexType.crossdoc:
            modes = ["L"]
        if t_vertex.type == VertexType.train:
            modes = ["L", "R"]
        if t_vertex.type == VertexType.port:
            modes = ["L", "S"]
        for mode in modes:
            t_node.capacity_handling["inc"][mode] = capacity
            t_node.capacity_handling["out"][mode] = capacity
            t_node.capacity_handling["tot"][mode] = 2 * capacity


def set_handling_capacities(warehouse_node_m: Dict[int, Node], 
                            transshipment_node_m: Dict[int, Node], 
                            capacitated: str,
                            r_demand_m: Dict[str, Dict[int, int]],
                            k_l: Dict[int, Commodity],
                            r_m: Dict[str, Region], 
                            v_m: Dict[int, Vertex],
                            d_l: Dict[int, Dict[int, int]]) -> None:
    """ Set handling capacities of warehouses and transshipment nodes. """
    _set_handling_capacities_warehouses(warehouse_node_m)
    if capacitated == "tight" or capacitated == "T":
        _set_handling_capacities_transshipment(transshipment_node_m, 
                                               r_demand_m, k_l, r_m, v_m, d_l)


def _start_costs(cy, cf, start_y):
    """ Return start costs given cy and cf are the discounted price rates. """
    cost_y = cy * start_y
    cost_f = cf * container_weight * (start_y - 1)
    costs = cost_y + cost_f
    return costs


def _gen_t_levels(cy, cf, start_y_l, discount_l):
    precision = 6
    t_levels = [TariffLevel(cy, cf, 0, 0)]
    # Additional all-unit discount levels
    for start_y, discount in zip(start_y_l, discount_l):
        cy_new = round(cy * (1 - discount), precision)
        cf_new = cf * (1 - discount) 
        b = round(_start_costs(cy_new, cf_new, start_y), precision)
        t_levels.append(TariffLevel(cy_new, cf_new, b, start_y))
    return t_levels


def _gen_lorry_tariffs(start_y_l: List[int], 
                       discount_l: List[float]) -> Tuple[Tariff, Tariff]:
    # Dry 
    cf = 0.005 / wbu_factor
    cy = 0.8
    t_levels = _gen_t_levels(cy, cf, start_y_l, discount_l)
    t_l = Tariff(0, [t_level for t_level in t_levels], "W")
    # Fresh 
    cy = 0.838
    t_levels = _gen_t_levels(cy, cf, start_y_l, discount_l)
    t_l_fresh = Tariff(1, [t_level for t_level in t_levels], "W")
    return t_l, t_l_fresh


def _gen_rail_tariffs(start_y_l: List[int], 
                      discount_l: List[float]) -> Tuple[Tariff, Tariff]:
    # Dry
    cf = 0
    cy = 0.5
    t_levels = _gen_t_levels(cy, cf, start_y_l, discount_l)
    t_r = Tariff(2, [t_level for t_level in t_levels], "W")
    # Fresh 
    cy = 0.553
    t_levels = _gen_t_levels(cy, cf, start_y_l, discount_l)
    t_r_fresh = Tariff(3, [t_level for t_level in t_levels], "W")
    return t_r, t_r_fresh


def _gen_ship_tariffs(start_y_l: List[int], 
                      discount_l: List[float]) -> Tuple[Tariff, Tariff]:
    # Dry
    cf = 0
    cy = 0.3
    t_levels = _gen_t_levels(cy, cf, start_y_l, discount_l)
    t_s = Tariff(4, [t_level for t_level in t_levels], "W")
    # Fresh 
    cy = 0.436
    t_levels = _gen_t_levels(cy, cf, start_y_l, discount_l)
    t_s_fresh = Tariff(5, [t_level for t_level in t_levels], "W")
    return t_s, t_s_fresh


def get_transport_tariffs(c_types: List[str]) \
                            -> Tuple[List[Tariff], List[Tariff], List[Tariff]]:
    """ Return transportation tariffs matching commodity types in instance. 
    
    Actually returned are lists of tariffs. The length of each list is deter-
    mined by the number of commodity types.
    """
    start_y_l = [5, 10]
    discount_l = [0.1, 0.2]
    t_l, t_l_fresh = _gen_lorry_tariffs(start_y_l, discount_l)
    t_r, t_r_fresh = _gen_rail_tariffs(start_y_l, discount_l)
    t_s, t_s_fresh = _gen_ship_tariffs(start_y_l, discount_l)
    # Return only tariffs for commodity types used in instance
    if len(c_types) == 2:
        t_l = [t_l, t_l_fresh]
        t_r = [t_r, t_r_fresh]
        t_s = [t_s, t_s_fresh]
    elif "F" in c_types:
        t_l = [t_l_fresh]
        t_r = [t_r_fresh]
        t_s = [t_s_fresh]
    else:
        t_l = [t_l]
        t_r = [t_r]
        t_s = [t_s]
    return t_l, t_r, t_s


def get_other_tariffs(c_types: List[str]) \
                            -> Tuple[List[Tariff], List[Tariff], List[Tariff]]:
    """ Return tariffs for storage and cost-free arcs. 
    
    Actually returned are lists of tariffs. The length of each list is deter-
    mined by the number of commodity types.
    """
    # Dry
    c_f = 0.16 / vbu_factor
    t_storage_lvl = TariffLevel(0, c_f, 0, 0)
    t_storage = Tariff(6, [t_storage_lvl], "V")
    # Fresh
    c_f = 0.32 / vbu_factor
    t_storage_lvl = TariffLevel(0, c_f, 0, 0)
    t_storage_fresh = Tariff(7, [t_storage_lvl], "V")
    # Free
    t_free_lvl = TariffLevel(0, 0, 0, 0)
    t_free = Tariff(8, [t_free_lvl], "V")
    # Return only tariffs for commodity types used in instance
    if len(c_types) == 2:
        t_storage = [t_storage, t_storage_fresh]
        t_free = [t_free]*2
    elif "F" in c_types:
        t_storage = [t_storage_fresh]
        t_free = [t_free]
    else:
        t_storage = [t_storage]
        t_free = [t_free]
    return t_storage, t_free


def to_list(t_l, t_r, t_s, t_storage, t_free) -> List[Tariff]:
    """ From list of tariff-lists, creates one list of tariffs. """
    tariff_ll = [t_l, t_r, t_s, t_storage, t_free]
    return [tariff for tariff_l in tariff_ll for tariff in tariff_l]


def reset_tariff_ids(tariff_l: List[Tariff]):
    """ Sets tariff ids to tariff-position in tariff_l. """
    for new_idx, tariff in enumerate(tariff_l):
        tariff.id = new_idx


def gen_instance(T, k_l, tariff_l: List[Tariff], 
                 warehouse_node_m, transshipment_node_m,
                 demand_node_m, bin_n, capacitated) -> Instance:
    """ Generate and return instance object. """
    c_type_l = _get_commodity_types(k_l)
    ins = Instance(
                ["single_tariff", 
                "multi_source", # multiple nodes can be source of same commodity
                ], 
                {"time_periods": T, 
                "c_types": c_type_l, "c_types_n": len(c_type_l),
                "c_properties": ["V", "W"], "c_properties_n": 2,
                "n_nodes_base": len(demand_node_m) + len(warehouse_node_m) 
                              + len(transshipment_node_m) + len([bin_n]), 
                "weight_cost": 1, # change in c++ programm
                "weight_green": 0, # change in c++ programm
                "transport_modes": ["L", "R", "S"], 
                "storage_mode": ["C"],
                "co2_costs_per_g": 0.0001} # corresponds to 100€/tCO2e
    )
    if capacitated == "tight" or capacitated == "T":
        ins.instance_properties.append( "capacitated" )
    ins.commodities.extend(k_l)
    ins.tariffs.extend(tariff_l)
    # Append nodes
    for key, node in warehouse_node_m.items():
        ins.nodes.append(node)
    for key, node in transshipment_node_m.items():
        ins.nodes.append(node)
    for key, node in demand_node_m.items():
        ins.nodes.append(node)
    ins.nodes.append(bin_n)
    return ins


def _get_emissions(c_types):
    """ Return emissions dictionaries for transport modes, storage and free. """
    # Decide on emission values to use depending on commodity types in instance
    start_idx = 0
    end_idx = 2
    if len(c_types) == 1:
        if c_types[0] == "N":
            end_idx = 1
        elif c_types[0] == "F":
            start_idx = 1
        else:
            raise NotImplementedError("ERROR in _get_emissions(): Only "
                    +"Emission values for dry (N) and fresh (F) products "
                    +f"supported! Found unknown type: {c_types[0]}")
    # Create emissions dictionaries
    ef = 69 / wbu_factor
    emissions_l = {"emissions_ykm": [933, 1011][start_idx:end_idx], 
                   "emissions_fkm": [ef, ef][start_idx:end_idx]}
    ef = 23 / wbu_factor
    emissions_r = {"emissions_ykm": [92, 201][start_idx:end_idx], 
                   "emissions_fkm": [ef, ef][start_idx:end_idx]}
    ef = 29 / wbu_factor
    emissions_s = {"emissions_ykm": [116, 396][start_idx:end_idx], 
                   "emissions_fkm": [ef, ef][start_idx:end_idx]}
    ef_dry = 16 / vbu_factor
    ef_fresh = 37 / vbu_factor
    emissions_storage = {"emissions_ykm": [0, 0][start_idx:end_idx], 
                         "emissions_fkm": [ef_dry, ef_fresh][start_idx:end_idx]}
    emission_free = {"emissions_ykm": [0, 0][start_idx:end_idx], 
                     "emissions_fkm": [0, 0][start_idx:end_idx]}
    return emissions_l, emissions_r, emissions_s, emissions_storage, \
           emission_free


def _get_capacities(c_types):
    """ Return TU capacities and uncapacitated upper bound on TU numbers. """
    uncapacitated = {}
    iso_cont_cap = {}
    storage_cont_cap = {}
    for c_type in c_types:
        # Define Upper Bound on Transport Unit Numbers
        uncapacitated[c_type] = -1
        # Define Transport Unit Capacities
        iso_cont_cap[c_type] = {"V": container_volume, "W": container_weight} 
        storage_cont_cap[c_type] = {"V": 1, "W": 1}
    return uncapacitated, iso_cont_cap, storage_cont_cap


def _gen_base_storage_arcs(arc_b_l: List[ArcBase], 
                           warehouse_node_m: Dict[int, Node], 
                           emissions_storage: Dict[str, Union[int, float]],
                           storage_cont_cap: Dict[str, Union[int, float]],
                           uncapacitated: Dict[str, Union[int, float]],
                           t_storage: List[Tariff]) -> None:
    """ Generates |warehouse nodes| arcs. 
    
    Assumes storage capacity is fixed for all time periods.
    """
    a_id = 0
    for _, n in warehouse_node_m.items():
        capacity_storage = {}
        for c_type, storage_capacity_l in n.capacity_storage.items():
            capacity_storage[c_type] = storage_capacity_l[0]
        arc_b_l.append(
            ArcBase(a_id, n, n, "C", 1, 1, t_storage, storage_cont_cap, 
                    capacity_storage, **emissions_storage)
        )
        a_id += 1


def _gen_base_own_demand_arcs(arc_b_l: List[ArcBase], 
                              warehouse_node_m: Dict[int, Node], 
                              transshipment_node_m: Dict[int, Node],
                              demand_node_m: Dict[int, Node], 
                              emission_free: Dict[str, Union[int, float]], 
                              cont_cap: Dict[str, Union[int, float]], 
                              uncapacitated: Dict[str, Union[int, float]], 
                              t_free: List[Tariff],
                              r_m: Dict[str, Region],
                              v_m: Dict[int, Vertex]) -> None:
    """ Generates |warehouse nodes| + O(transshipment nodes) arcs. """
    a_id = len(arc_b_l)
    # Generate arc to regional demand node for each warehouse
    for w_n_idx in warehouse_node_m:
        # Get associated demand node idx for warehouse node
        d_n_idx = r_m[v_m[w_n_idx].region].idx
        arc_b_l.append(
            ArcBase(a_id, w_n_idx, d_n_idx, "O", 0, 0, t_free, cont_cap, 
                    uncapacitated, **emission_free)
        )
        a_id += 1
    # Genereate arc if regional demand node exists for each transshipment point
    for t_n_idx in transshipment_node_m:
        region = v_m[t_n_idx].region
        if region in r_m: 
            d_n_idx = r_m[region].idx
            arc_b_l.append(
                ArcBase(a_id, t_n_idx, d_n_idx, "O", 0, 0, t_free, cont_cap, 
                        uncapacitated, **emission_free)
            )
            a_id += 1


def _gen_base_inter_warehouse_arcs(arc_b_l, d_l, t_matrix_l,
                                   warehouse_node_m, emissions_l,
                                   iso_cont_cap, uncapacitated, t_l,
                                   direct_delivery, group_w_l):
    """ Generates |warehouse nodes|*(|warehouse nodes|-1) arcs. 
    
    If direct delivery is true, only generates inter-warehouse arcs between
    warehouses of the same group.
    """       
    emissions_yh = 0
    emissions_fh = 510 * 2 / wbu_factor
    cost_yh = 0
    cost_fh = 2 / wbu_factor

    a_id = len(arc_b_l)
    # Connect each warehouse with each other except oneself
    for w_idx1 in warehouse_node_m:
        if direct_delivery:
            group_idx = -1
            for i, group in enumerate(group_w_l):
                if w_idx1 in group:
                    group_idx = i
                    break
        for w_idx2 in warehouse_node_m:
            if w_idx1 == w_idx2:
                continue
            if direct_delivery:
                if w_idx2 not in group_w_l[group_idx]:
                    continue
            d = d_l[w_idx1][w_idx2]
            t = t_matrix_l[w_idx1][w_idx2]
            arc_b_l.append(
                ArcBase(a_id, w_idx1, w_idx2, "L", d, t, t_l, iso_cont_cap, 
                        uncapacitated, **emissions_l, emissions_yh=emissions_yh, 
                        emissions_fh=emissions_fh, cost_yh=cost_yh, 
                        cost_fh=cost_fh)
            )
            a_id += 1


def _get_handling_costs_emissions(t1: Vertex, t2: Vertex, mode: str="L"):
    """ Given two vertices being facilities returns handling costs & emissions.
    
    Returns in that order cost_yh, cost_fh, emissions_yh, emissions_fh.
    """
    e_yh_crossdock = 0
    e_fh_crossdock = 255 / wbu_factor
    e_yh_w = 0
    e_fh_w = 510 / wbu_factor
    e_yh_rail = 6289
    e_fh_rail = 0 / wbu_factor
    e_yh_ship = 6984 
    e_fh_ship = 1746 / wbu_factor

    cost_yh = 0
    cost_fh_crossdock = 0.5 / wbu_factor
    cost_fh_rs = 1.5 / wbu_factor #only one of two operations required!
    cost_fh_w = 1 / wbu_factor

    if mode == "L":
        if t1.type == VertexType.warehouse and t2.type != VertexType.demand:
            # Warehouse -> Transshipment
            if t2.type == VertexType.crossdoc:
                cost_fh = cost_fh_w + cost_fh_crossdock
                emissions_yh = e_yh_crossdock + e_yh_w
                emissions_fh = e_fh_crossdock + e_fh_w
            elif t2.type == VertexType.train:
                cost_fh = cost_fh_w + cost_fh_rs
                emissions_yh = e_yh_w + e_yh_rail
                emissions_fh = e_fh_w + e_fh_rail
            elif t2.type == VertexType.port:
                cost_fh = cost_fh_w + cost_fh_rs
                emissions_yh = e_yh_w + e_yh_ship
                emissions_fh = e_fh_w + e_fh_ship
            else:
                raise RuntimeError("_get_handling_costs_emissions(): Ups, should "
                                +"have never reached this one!")
        elif t1.type in [VertexType.crossdoc, VertexType.train, VertexType.port] \
                and \
            t2.type in [VertexType.crossdoc, VertexType.train, VertexType.port]:
            # Transshipment <-> Transshipment
            if t1.type == t2.type:
                # Case 1: cross dock <-> cross dock
                if t1.type == VertexType.crossdoc:
                    cost_fh = cost_fh_crossdock * 2
                    emissions_yh = e_yh_crossdock * 2
                    emissions_fh = e_fh_crossdock * 2
                # Case 2: rail <-> rail
                if t1.type == VertexType.train:
                    cost_fh = cost_fh_rs * 2
                    emissions_yh = e_yh_rail * 2
                    emissions_fh = e_fh_rail * 2
                # Case 3: ship <-> ship
                if t1.type == VertexType.port:
                    cost_fh = cost_fh_rs * 2
                    emissions_yh = e_yh_ship * 2
                    emissions_fh = e_fh_ship * 2
            # cross dock <-> train station
            elif (t1.type == VertexType.crossdoc and t2.type == VertexType.train) \
                    or \
                (t2.type == VertexType.crossdoc and t1.type == VertexType.train):
                cost_fh = cost_fh_crossdock + cost_fh_rs
                emissions_yh = e_yh_crossdock + e_yh_rail
                emissions_fh = e_fh_crossdock + e_fh_rail
            # cross dock <-> port
            elif (t1.type == VertexType.crossdoc and t2.type == VertexType.port) \
                    or \
                (t2.type == VertexType.crossdoc and t1.type == VertexType.port):
                cost_fh = cost_fh_crossdock + cost_fh_rs
                emissions_yh = e_yh_crossdock + e_yh_ship
                emissions_fh = e_fh_crossdock + e_fh_ship
            # train stations <-> port
            elif (t1.type == VertexType.train and t2.type == VertexType.port) \
                    or \
                (t2.type == VertexType.train and t1.type == VertexType.port):
                cost_fh = cost_fh_rs + cost_fh_rs
                emissions_yh = e_yh_rail + e_yh_ship
                emissions_fh = e_fh_rail + e_fh_ship
            else:
                raise RuntimeError("_get_handling_costs_emissions(): Ups, this part"
                                +" of the code should have never been reached.")
        elif t2.type == VertexType.demand and t1.type != VertexType.demand:
            # facility <-> external demand node
            if t1.type == VertexType.warehouse:
                cost_fh = cost_fh_w
                emissions_yh = e_yh_w
                emissions_fh = e_fh_w
            elif t1.type == VertexType.crossdoc:
                cost_fh = cost_fh_crossdock
                emissions_yh = e_yh_crossdock  
                emissions_fh = e_fh_crossdock  
            elif t1.type == VertexType.train:
                cost_fh = cost_fh_rs
                emissions_yh = e_yh_rail
                emissions_fh = e_fh_rail
            elif t1.type == VertexType.port:
                cost_fh = cost_fh_rs
                emissions_yh = e_yh_ship
                emissions_fh = e_fh_ship
            else:
                raise RuntimeError("_get_handling_costs_emissions(): Ups, you should"
                                +" have never went here.")
        else:
            raise NotImplementedError("_get_handling_costs_emissions(): Vertex " 
                                    +f"type combination {t1.type}, {t2.type} "
                                    +f"for mode L not supported.")
    elif mode == "R":
        if t1.type == VertexType.train and t2.type == VertexType.train:
            cost_fh = cost_fh_rs + cost_fh_rs
            emissions_yh = e_yh_rail * 2
            emissions_fh = e_fh_rail * 2
        else:
            raise RuntimeError("_get_handling_costs_emissions(): A train can "
                              +"can only drive between train-stations.")
    elif mode == "S":
        if t1.type == VertexType.port and t2.type == VertexType.port:
            cost_fh = cost_fh_rs + cost_fh_rs
            emissions_yh = e_yh_ship * 2
            emissions_fh = e_fh_ship * 2
        else:
            raise RuntimeError("_get_handling_costs_emissions(): A ship can "
                              +"can only float happily between ports.")
    else:
        raise NotImplementedError(f"_get_handling_costs_emissions(): mode "
                                 +f"{mode} not supported.")                         
    return cost_yh, cost_fh, emissions_yh, emissions_fh


def _gen_base_warehouse_transshipment_arcs(arc_b_l, d_l, t_matrix_l,
                                warehouse_node_m, transshipment_node_m,
                                emissions_l, iso_cont_cap, uncapacitated, t_l,
                                v_m):
    """ Connect each warehouse with each transshipment facility via truck. 
    
    Add |warehouses|*|transshipment| arcs.
    """
    a_id = len(arc_b_l)
    for w_idx in warehouse_node_m:
        w_n = v_m[w_idx]
        for t_idx in transshipment_node_m:
            # Get handling costs & emissions
            t_n = v_m[t_idx]
            cost_yh, cost_fh, emissions_yh, emissions_fh = \
                                        _get_handling_costs_emissions(w_n, t_n)
            # Warehouse -> Transshipment Arc
            d = d_l[w_idx][t_idx]
            t = t_matrix_l[w_idx][t_idx]
            # Create Arc
            arc_b_l.append(
                ArcBase(a_id, w_idx, t_idx, "L", d, t, t_l, iso_cont_cap, 
                        uncapacitated, **emissions_l, 
                        emissions_yh=emissions_yh, 
                        emissions_fh=emissions_fh, cost_yh=cost_yh, 
                        cost_fh=cost_fh)
            )
            a_id += 1


def _gen_base_inter_transshipment_arcs(arc_b_l, d_l, t_matrix_l,
                                transshipment_node_m, emissions_l, 
                                iso_cont_cap, uncapacitated, t_l, v_m):
    """ Connect each transshipment with each transshipment node via truck. 
    
    Add |transshipment|*(|transshipment|-1) arcs.
    """
    a_id = len(arc_b_l)
    for t1_idx in transshipment_node_m:
        t1 = v_m[t1_idx]
        for t2_idx in transshipment_node_m:
            if t1_idx == t2_idx:
                continue
            d = d_l[t1_idx][t2_idx]
            t = t_matrix_l[t1_idx][t2_idx]
            # Set handling costs & emissions based transshipment connection
            t2 = v_m[t2_idx]
            cost_yh, cost_fh, emissions_yh, emissions_fh = \
                                        _get_handling_costs_emissions(t1, t2)
            # Create Arc
            arc_b_l.append(
                ArcBase(a_id, t1_idx, t2_idx, "L", d, t, t_l, iso_cont_cap, 
                        uncapacitated, **emissions_l, 
                        emissions_yh=emissions_yh, 
                        emissions_fh=emissions_fh, cost_yh=cost_yh, 
                        cost_fh=cost_fh)
            )
            a_id += 1
                        

def _gen_base_transshipment_warehouse_arcs(arc_b_l, d_l, t_matrix_l,
                                warehouse_node_m, transshipment_node_m,
                                emissions_l, iso_cont_cap, uncapacitated, t_l,
                                v_m):
    """ Connect each transshipment facility with each warehouse via truck. 
    
    Very much related to _gen_base_warehouse_transshipment_arcs(). 

    Add |warehouses|*|transshipment| arcs.
    """
    a_id = len(arc_b_l)
    for t_idx in transshipment_node_m:
        t_n = v_m[t_idx]
        for w_idx in warehouse_node_m:
            # Get (symmetric!) handling costs & emissions
            w_n = v_m[w_idx]
            cost_yh, cost_fh, emissions_yh, emissions_fh = \
                                        _get_handling_costs_emissions(w_n, t_n)
            # Warehouse -> Transshipment Arc
            d = d_l[t_idx][w_idx]
            t = t_matrix_l[t_idx][w_idx]
            # Create Arc
            arc_b_l.append(
                ArcBase(a_id, t_idx, w_idx, "L", d, t, t_l, iso_cont_cap, 
                        uncapacitated, **emissions_l, 
                        emissions_yh=emissions_yh, 
                        emissions_fh=emissions_fh, cost_yh=cost_yh, 
                        cost_fh=cost_fh)
            )
            a_id += 1


def _gen_base_external_demand_arcs(arc_b_l, d_l, t_matrix_l, warehouse_node_m,
                                   transshipment_node_m, demand_node_m,
                                   emissions_l, iso_cont_cap, uncapacitated,
                                   t_l, r_m, v_m):
    """ Connect each facility with all external demand regions (nodes). 
    
    Add (|warehouses|+|transshipment|)*O(|demand_regions|) arcs.
    Note: Mostly should be (|warehouses|+|transshipment|)*(|demand_regions|-1)
          arcs. But if a transshipment node is not in one of the demand regions
          it will be connected to all demand regions. 
    """
    a_id = len(arc_b_l)
    for f_idx in list(transshipment_node_m) + list(warehouse_node_m):
        f_n = v_m[f_idx]
        for d_region in demand_node_m:
            if d_region == f_n.region:
                continue
            d_idx = r_m[d_region].idx
            d_n = v_m[d_idx]
            cost_yh, cost_fh, emissions_yh, emissions_fh = \
                                        _get_handling_costs_emissions(f_n, d_n)
            # Create Arc
            d = d_l[f_idx][d_idx]
            t = t_matrix_l[f_idx][d_idx]
            arc_b_l.append(
                ArcBase(a_id, f_idx, d_idx, "L", d, t, t_l, iso_cont_cap, 
                        uncapacitated, **emissions_l, 
                        emissions_yh=emissions_yh, 
                        emissions_fh=emissions_fh, cost_yh=cost_yh, 
                        cost_fh=cost_fh)
            )
            a_id += 1


def _gen_base_lorry_arcs(arc_b_l: List[ArcBase], 
                         d_l: Dict[int, Dict[int, int]],
                         t_matrix_l: Dict[int, Dict[int, int]],
                         warehouse_node_m: Dict[int, Node], 
                         transshipment_node_m: Dict[int, Node],
                         demand_node_m: Dict[str, Node], 
                         emissions_l: Dict[str, List[int]], 
                         iso_cont_cap: Dict[str, float], 
                         uncapacitated: Dict[str, float],
                         t_l: List[Tariff],
                         r_m: Dict[str, Region],
                         v_m: Dict[int, Vertex],
                         direct_delivery: bool,
                         group_w_l: List[List[int]]) -> None:
    """ Generate all base-arcs of mode lorry. """
    _gen_base_inter_warehouse_arcs(arc_b_l, d_l, t_matrix_l,
                                warehouse_node_m, emissions_l,
                                iso_cont_cap, uncapacitated, t_l, 
                                direct_delivery, group_w_l)
    if not direct_delivery:
        _gen_base_warehouse_transshipment_arcs(arc_b_l, d_l, t_matrix_l,
                                    warehouse_node_m, transshipment_node_m,
                                    emissions_l, iso_cont_cap, uncapacitated, t_l,
                                    v_m)
        _gen_base_inter_transshipment_arcs(arc_b_l, d_l, t_matrix_l,
                                    transshipment_node_m, emissions_l, 
                                    iso_cont_cap, uncapacitated, t_l, v_m)
        _gen_base_transshipment_warehouse_arcs(arc_b_l, d_l, t_matrix_l,
                                    warehouse_node_m, transshipment_node_m,
                                    emissions_l, iso_cont_cap, uncapacitated, t_l,
                                    v_m)
    _gen_base_external_demand_arcs(arc_b_l, d_l, t_matrix_l, warehouse_node_m,
                                   transshipment_node_m, demand_node_m,
                                   emissions_l, iso_cont_cap, uncapacitated,
                                   t_l, r_m, v_m)


def _gen_base_rail_arcs(arc_b_l, d_r, t_matrix_r,
                        transshipment_node_m,
                        emissions_r, iso_cont_cap, uncapacitated, t_r, v_m):
    """ Connect each train-station with one another by train. 
    
    Add (|train stations|)*(|train stations|-1) arcs.
    """
    a_id = len(arc_b_l)
    for f1_idx in transshipment_node_m:
        f1_n = v_m[f1_idx]
        if f1_n.type != VertexType.train:
            continue
        for f2_idx in transshipment_node_m:
            f2_n = v_m[f2_idx]
            if f2_n.type != VertexType.train:
                continue
            if f2_idx == f1_idx:
                continue
            cost_yh, cost_fh, emissions_yh, emissions_fh = \
                                _get_handling_costs_emissions(f1_n, f2_n, "R")
            # Create Arc
            d = d_r[f1_idx][f2_idx]
            t = t_matrix_r[f1_idx][f2_idx]
            arc_b_l.append(
                ArcBase(a_id, f1_idx, f2_idx, "R", d, t, t_r, iso_cont_cap, 
                        uncapacitated, **emissions_r, 
                        emissions_yh=emissions_yh, 
                        emissions_fh=emissions_fh, cost_yh=cost_yh, 
                        cost_fh=cost_fh)
            )
            a_id += 1


def _gen_base_ship_arcs(arc_b_l, d_s, t_matrix_s,
                        transshipment_node_m,
                        emissions_s, iso_cont_cap, uncapacitated, t_s, v_m):
    """ Connect each port with one another by ship. 
    
    Add (|ports|)*(|ports|-1) arcs.
    """
    a_id = len(arc_b_l)
    for f1_idx in transshipment_node_m:
        f1_n = v_m[f1_idx]
        if f1_n.type != VertexType.port:
            continue
        for f2_idx in transshipment_node_m:
            f2_n = v_m[f2_idx]
            if f2_n.type != VertexType.port:
                continue
            if f2_idx == f1_idx:
                continue
            cost_yh, cost_fh, emissions_yh, emissions_fh = \
                                _get_handling_costs_emissions(f1_n, f2_n, "S")
            # Create Arc
            d = d_s[f1_idx][f2_idx]
            t = t_matrix_s[f1_idx][f2_idx]
            arc_b_l.append(
                ArcBase(a_id, f1_idx, f2_idx, "S", d, t, t_s, iso_cont_cap, 
                        uncapacitated, **emissions_s, 
                        emissions_yh=emissions_yh, 
                        emissions_fh=emissions_fh, cost_yh=cost_yh, 
                        cost_fh=cost_fh)
            )
            a_id += 1


def _gen_base_bin_arcs(arc_b_l, warehouse_node_m, transshipment_node_m, bin_n,
                       emission_free, storage_cont_cap, uncapacitated, t_free):
    """ Connect each facility with the bin node. 
    
    Add |warehouses| + |transshipment nodes| arcs.
    """
    a_id = len(arc_b_l)
    for f_idx in list(transshipment_node_m) + list(warehouse_node_m):
        # Create Arc
        d = 0
        t = 0
        arc_b_l.append(
            ArcBase(a_id, f_idx, bin_n, "O", d, t, t_free, storage_cont_cap, 
                    uncapacitated, **emission_free, 
                    emissions_yh=0, emissions_fh=0, 
                    cost_yh=0, cost_fh=0)
        )
        a_id += 1


def gen_base_arcs(d_l: Dict[int, Dict[int, int]], d_r, d_s,
                   t_matrix_l: Dict[int, Dict[int, int]], t_matrix_r, t_matrix_s,
                   t_l: List[Tariff], t_r, t_s, t_storage, t_free,
                   warehouse_node_m: Dict[int, Node], 
                   transshipment_node_m: Dict[int, Node], 
                   demand_node_m: Dict[str, Node],
                   bin_n: Node, 
                   c_types: List[str],
                   r_m: Dict[str, Region],
                   v_m: Dict[int, Vertex],
                   direct_delivery: bool,
                   group_w_l: List[List[int]]) -> List[ArcBase]:
    """ Generate base arcs of network. """
    # Generate necessary emission and capacity data (dictionaries)
    emissions_l, emissions_r, emissions_s, emissions_storage, emission_free \
                                                    = _get_emissions(c_types)
    uncapacitated, iso_cont_cap, storage_cont_cap = _get_capacities(c_types) 
    # Generate and fill base arc list
    arc_b_l = []
    _gen_base_storage_arcs(arc_b_l, warehouse_node_m, emissions_storage,
                           storage_cont_cap, uncapacitated, t_storage)
    if not direct_delivery:
        _gen_base_own_demand_arcs(arc_b_l, warehouse_node_m, transshipment_node_m,
                                demand_node_m, emission_free, storage_cont_cap, 
                                uncapacitated, t_free, r_m, v_m)
    _gen_base_lorry_arcs(arc_b_l, d_l, t_matrix_l,
                        warehouse_node_m, transshipment_node_m,
                        demand_node_m, emissions_l, iso_cont_cap, 
                        uncapacitated, t_l, r_m, v_m, 
                        direct_delivery, group_w_l)
    if not direct_delivery:
        _gen_base_rail_arcs(arc_b_l, d_r, t_matrix_r,
                            transshipment_node_m,
                            emissions_r, iso_cont_cap, uncapacitated, t_r, v_m)
        _gen_base_ship_arcs(arc_b_l, d_s, t_matrix_s,
                            transshipment_node_m,
                            emissions_s, iso_cont_cap, uncapacitated, t_s, v_m)
    _gen_base_bin_arcs(arc_b_l, warehouse_node_m, transshipment_node_m, bin_n,
                    emission_free, storage_cont_cap, uncapacitated, t_free)
    return arc_b_l


def reset_node_ids(ins: Instance, arc_b_l: List[ArcBase]):
    """ Create 0-based numbering of nodes in ins and adapt arcs in arc_b_l. """
    old_to_new_id = {}
    for new_id, node in enumerate(ins.nodes):
        node.data = node.id
        old_to_new_id[node.id] = new_id
        node.id = new_id
    for arc in arc_b_l:
        arc.node_orig = old_to_new_id[arc.node_orig]
        arc.node_dest = old_to_new_id[arc.node_dest]
    return old_to_new_id


def gen_time_expanded_arcs(T: int, arc_b_l: List[ArcBase]):
    """ Return time expantion for T periods of base arcs in arc_b_l. """
    arc_l = []
    idx = 0
    for arc in arc_b_l:
        for t in range(T):
            if t + arc.time < T:
                arc_l.append(Arc(idx, arc, t, t+arc.time))
                idx += 1
    return arc_l


def _get_closest_warehouse_with_k(k, d_l, d_n, warehouse_node_m):
    min_d = -1
    min_w_idx = -1
    for w_idx, w_n in warehouse_node_m.items():
        if k in w_n.stocks:
            d = d_l[w_idx][d_n.data]
            if d < min_d or min_d == -1:
                min_d = d
                min_w_idx = w_idx
    return min_w_idx


def check_feasible(d_l, t_matrix_l, warehouse_node_m, demand_node_m, 
                   t_l, r_m, v_m, k_l, arc_l):
    """ Check whether stock-wise and handling-capacity wise, generated isntance
        can have a solution. """
    w_stock_d = {}
    w_handling_d = {}
    v_handling_d = {}
    max_w_shipped = 0
    for region, d_n in demand_node_m.items():
        for k, stock in d_n.stocks.items():
            # Check if demand can be satisfied and handled!
            closest_w_idx = _get_closest_warehouse_with_k(k, d_l, d_n,
                                                          warehouse_node_m)
            if closest_w_idx not in w_stock_d:
                w_stock_d[closest_w_idx] = {}
                w_handling_d[closest_w_idx] = {}
                v_handling_d[closest_w_idx] = {}
                for t in range(len(stock)):
                    w_handling_d[closest_w_idx][t] = \
                        copy.deepcopy(warehouse_node_m[closest_w_idx].capacity_handling)
                    v_handling_d[closest_w_idx][t] = \
                        copy.deepcopy(warehouse_node_m[closest_w_idx].capacity_handling)
                    if w_handling_d[closest_w_idx] == {} or \
                            w_handling_d[closest_w_idx][t]['out']['L'] == -1:
                        w_handling_d[closest_w_idx][t] = "Unbounded"
                    else:
                        w_handling_d[closest_w_idx][t]['out']['L'] *= container_weight
                        w_handling_d[closest_w_idx][t]['tot']['L'] *= container_weight
                        v_handling_d[closest_w_idx][t]['out']['L'] *= container_volume
                        v_handling_d[closest_w_idx][t]['tot']['L'] *= container_volume
            if k not in w_stock_d[closest_w_idx]:
                w_stock_d[closest_w_idx][k] = \
                    [s for s in warehouse_node_m[closest_w_idx].stocks[k]]
            w_stock = w_stock_d[closest_w_idx][k]
            t_min = t_matrix_l[closest_w_idx][d_n.data]
            # Preprocess to find arc fast
            warehouse = warehouse_node_m[closest_w_idx]
            possible_aidx_l = []
            for idx, arc in enumerate(arc_l):
                if arc.node_orig == warehouse.id and \
                    arc.node_dest == d_n.id:
                    possible_aidx_l.append(idx)
            # Check if every time period satisfyable
            for t, s in enumerate(stock):
                if t - t_min < 0 and s < 0:
                    print("Infeasibility detected!")
                elif t - t_min < 0:
                    continue
                else:
                    # find arc
                    necessary_arc_exists = False
                    for aidx in possible_aidx_l:
                        arc = arc_l[aidx]
                        if arc.t_start == t - t_min and \
                            arc.t_end == t:
                            necessary_arc_exists = True
                            break
                    if not necessary_arc_exists:
                        print("There is no arc to directly fullfill demand!")
                    # Check satisfyabilit by stock-values and handling capacity
                    w_stock[t - t_min] = w_stock[t - t_min] + s
                    if w_stock[t - t_min] < 0:
                        print("Infeasibility detected!")
                    if w_handling_d[closest_w_idx][t] != "Unbounded":
                        w_shipped = -1 * s * k_l[k].properties["W"]
                        v_shipped = -1 * s * k_l[k].properties["V"]
                        if w_shipped > max_w_shipped:
                            max_w_shipped = w_shipped
                        w_handling_d[closest_w_idx][t - t_min]['out']['L'] -= w_shipped
                        w_handling_d[closest_w_idx][t - t_min]['tot']['L'] -= w_shipped
                        v_handling_d[closest_w_idx][t - t_min]['out']['L'] -= v_shipped
                        v_handling_d[closest_w_idx][t - t_min]['tot']['L'] -= v_shipped
                        if w_handling_d[closest_w_idx][t - t_min]['out']['L'] < 0 or \
                                w_handling_d[closest_w_idx][t - t_min]['tot']['L'] < 0 or \
                                v_handling_d[closest_w_idx][t - t_min]['out']['L'] < 0 or \
                                v_handling_d[closest_w_idx][t - t_min]['tot']['L'] < 0:
                            print("Infeasible by not enough handling cap.")
            # Calulate price of demand satisfaction
            # TODO
    pass


def gen_random_instance(C: int, R: List[int], use_regions: List[str], 
                        W: List[int], K: int, 
                        T: int, F: int, modes: List[str], capacitated: str, 
                        max_demand: int,
                        direct_delivery: bool, seed: int, 
                        distance_matrix_f: str, time_matrix_f: str, 
                        locations_f: str) -> Instance:
    np.random.seed(seed)
    random.seed(seed)
    if modes is not None:
        global mode_l
        mode_l = modes
    # Read in regions and nodes map
    # r_m key: region name (str), v_m key: node index in file
    r_m, v_m = read_locations(locations_f, W)
    max_vertex_id = get_max_vertex_id(v_m)
    # Choose user defined (number of) regions
    select_regions(r_m, v_m, R, use_regions)
    d_l, d_s, d_r = read_distance_matrix(distance_matrix_f, v_m)
    # Remove Transshipment Points not closet to any region in r_m
    select_closest_transshipment(r_m, v_m, d_l, d_s, d_r, direct_delivery)
    t_matrix_l, t_matrix_s, t_matrix_r = read_traveltime_matrix(time_matrix_f, 
                                                                v_m)
    correct_time_matrix_l(t_matrix_l, v_m, r_m)

    #### Concerning Nodes & Commodities

    # Init "Companies"
    group_w_l = [[] for el in range(0, C)]
    associate_warehouses_to_group(group_w_l, v_m)
    
    # Init Commodities List. Commodity position equals its id.
    k_l = init_commodities(K, F)
    # Define which groups hold stock of certain products
    group_k_l = [[] for el in range(0, C)]
    associate_products_to_group(group_k_l, group_w_l, k_l, F)
    # Define where demand for certain products exists
    k_r_m = {} # map product id to demand regions
    associate_products_to_demandregions(k_r_m, group_k_l, group_w_l, v_m, r_m)

    # Generate bin node with id = 1 + max vertex id in locations file
    bin_n = gen_bin_node(max_vertex_id, k_l)
    # Generate Demand Nodes without stock
    demand_node_m = gen_demand_nodes(r_m, k_l)
    # Generate Warehouse Nodes without stock
    warehouse_node_m = gen_warehouse_nodes(v_m, k_l)
    # Set stock of demand and warehouse nodes
    r_demand_m = set_stock(T, d_l, t_matrix_l, demand_node_m, warehouse_node_m, 
                           group_w_l, group_k_l, k_r_m, k_l, r_m, v_m, max_demand)
    # Set different commodity attributes
    set_lifetime(K, F, d_l, t_matrix_l, demand_node_m, group_w_l, group_k_l, 
                 k_r_m, k_l)
    set_commodity_group(group_k_l, k_l)
    # Set storage capacities for warehouses
    set_storage_capacities(T, warehouse_node_m, k_l, r_m, v_m)
    # Generate Transshipment Nodes & set handling capacities
    transshipment_node_m = gen_transshipment_nodes(v_m, k_l)
    set_handling_capacities(warehouse_node_m, transshipment_node_m, 
                            capacitated, r_demand_m, k_l, r_m, v_m, d_l)
    
    ### Concerning Arcs & Tariffs
    
    # Tariffs
    c_types = _get_commodity_types(k_l)
    t_l, t_r, t_s = get_transport_tariffs(c_types)
    t_storage, t_free = get_other_tariffs(c_types)
    tariff_l = to_list(t_l, t_r, t_s, t_storage, t_free)
    # Reset tariff ids to be zero based and incremental
    reset_tariff_ids(tariff_l)

    # Instance
    ins = gen_instance(T, k_l, tariff_l, warehouse_node_m, 
                       transshipment_node_m, demand_node_m, bin_n, capacitated)
    
    # Arcs
    arc_b_l = gen_base_arcs(d_l, d_r, d_s, t_matrix_l, t_matrix_r, t_matrix_s,
                             t_l, t_r, t_s, t_storage, t_free,
                             warehouse_node_m, transshipment_node_m, 
                             demand_node_m, bin_n, c_types, r_m, v_m,
                             direct_delivery, group_w_l)
    old_to_new_id = reset_node_ids(ins, arc_b_l)
    arc_l = gen_time_expanded_arcs(T, arc_b_l)
    ins.arcs.extend(arc_l)
    ins.other["n_arcs"] = len(arc_l)
    
    # Debug
    check_feasible(d_l, t_matrix_l, warehouse_node_m, demand_node_m, 
                   t_l, r_m, v_m, k_l, arc_l)

    return ins
