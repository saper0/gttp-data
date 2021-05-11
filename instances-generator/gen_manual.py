from instance import *

def gen_manual_instance(identifier: int) -> Instance:
    """Generate a manual instance based on given identifier. """
    if identifier == 1:
        return _gen_manual_instance_1()
    elif identifier == 2:
        return _gen_manual_instance_2()
    elif identifier == 3:
        return _gen_manual_instance_3()
    elif identifier == 4:
        return _gen_manual_instance_4()
    elif identifier == 5:
        return _gen_manual_instance_5()
    elif identifier == 6:
        return _gen_manual_instance_6()
    else:
        raise NotImplementedError("Only manual instance 1 to 6 implemented.")


def _gen_manual_instance_1() -> Instance:
    """Generate manual instance one. Single mode, single product. """

    k = Commodity(0, {"V": 1, "W": 1}, "N")

    # Nodes
    v1_b = NodeBase(0, "facility", {"N": 20}, {"inc": {"L": 5},
                    "out": {"L": 5}, "tot": {"L": 10}})
    v1 = Node(v1_b, {0: [11, 0]})
    v2_b = NodeBase(1, "facility", {"N": 20}, {"inc": {"L": 15},
                    "out": {"L": 15}, "tot": {"L": 15}})
    v2 = Node(v2_b, {0: [14, 0]})

    d1_b = NodeBase(2, "demand")
    d1 = Node(d1_b, {0: [0, -14]})
    d2_b = NodeBase(3, "demand")
    d2 = Node(d2_b, {0: [0, -10]})

    bin_b = NodeBase(4, "bin")
    bin_ = Node(bin_b)

    # Tariffs
    t_const_lvl1 = TariffLevel(1, 0, 0, 0)
    t_const_lvl2 = TariffLevel(0.9, 0, 2.7, 3)
    t_const_lvl3 = TariffLevel(0.8, 0, 4.8, 6)
    t_const_discount = Tariff(0, [t_const_lvl1, t_const_lvl2, t_const_lvl3], "W")

    t_lin_lvl1 = TariffLevel(1, 0.5, 0, 0)
    t_lin_lvl2 = TariffLevel(0.9, 0.5, 4.7, 3)
    t_lin_lvl3 = TariffLevel(0.8, 0.5, 8.0, 5)
    t_lin_discount = Tariff(1, [t_lin_lvl1, t_lin_lvl2, t_lin_lvl3], "W")

    t_const_lvl = TariffLevel(0, 0.01, 0, 0)
    t_const = Tariff(2, [t_const_lvl], "V")

    t_free_lvl = TariffLevel(0, 0, 0, 0)
    t_free = Tariff(3, [t_free_lvl], "V")

    # Instance
    ins = Instance(
                ["single_tariff", "no_perishable", "capacitated"], 
                {"time_periods": 2, 
                "c_types": ["N"], "c_types_n": 1,
                "c_properties": ["V", "W"], "c_properties_n": 2,
                "n_nodes_base": 5, "weight_cost": 1, "weight_green": 0,
                "transport_modes": ["L"], "storage_mode": ["C"],
                "co2_costs_per_g": 0.0001}
    )
    ins.commodities.extend([k])
    ins.tariffs.extend([t_const_discount, t_lin_discount, t_const, t_free])
    ins.nodes.extend([v1, v2, d1, d2, bin_])

    ## Arcs
    iso_cont_cap = {"N":{"V": 2, "W": 2}}
    str_cont_cap = {"N":{"V": 1, "W": 1}}
    uncapacitated = {"N": -1}

    # Transport
    a1_b = ArcBase(0, v1_b, v2_b, "L", 1, 1, t_const_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.04)
    a2_b = ArcBase(1, v2_b, v1_b, "L", 1, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.04)
    a1 = Arc(0, a1_b, 0, 1)
    a2 = Arc(1, a2_b, 0, 1)

    # Storage
    str1_b = ArcBase(2, v1_b, v1_b, "C", 1, 1, t_const, str_cont_cap, {"N": 20},
                        emissions_fkm=[1])
    str2_b = ArcBase(3, v2_b, v2_b, "C", 1, 1, t_const, str_cont_cap, {"N": 20},
                        emissions_fkm=[1])
    str1 = Arc(2, str1_b, 0, 1)
    str2 = Arc(3, str2_b, 0, 1)
    
    ins.arcs.extend([a1, a2, str1, str2])

    # Regional Demand
    da1_b = ArcBase(4, v1_b, d1_b, "O", 0, 0, t_free, str_cont_cap, uncapacitated)
    da2_b = ArcBase(5, v2_b, d2_b, "O", 0, 0, t_free, str_cont_cap, uncapacitated)
    idx = 3
    for t in [0, 1]:
        ins.arcs.append(Arc(idx+1, da1_b, t, t))
        ins.arcs.append(Arc(idx+2, da2_b, t, t))
        idx += 2
    
    # Other Demand
    dda1_b = ArcBase(6, v1_b, d2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.02)
    dda2_b = ArcBase(7, v2_b, d1_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.02)
    dda1 = Arc(idx+1, dda1_b, 0, 1)
    dda2 = Arc(idx+2, dda2_b, 0, 1)
    idx = idx + 2
    ins.arcs.extend([dda1, dda2])

    # Bin 
    facilities = [v1_b, v2_b]
    T = [0, 1]
    idx_b = 7
    for v in facilities:
        arc_base = ArcBase(idx_b + 1, v, bin_b, "O", 0, 0, t_free, str_cont_cap, 
                            uncapacitated)
        for t in T:
            idx += 1
            ins.arcs.append(Arc(idx, arc_base, t, t))
        idx_b += 1
    
    # Statistics
    ins.other["n_arcs"] = idx + 1

    return ins


def _gen_manual_instance_2() -> Instance:
    """Generate manual instance two. Single mode, two products, handling capacities. """

    k0 = Commodity(0, {"V": 1, "W": 1}, "N")
    k1 = Commodity(1, {"V": 1, "W": 1}, "N")

    # Nodes
    v1_b = NodeBase(0, "facility", {"N": 10}, {"inc": {"L": 5},
                    "out": {"L": 5}, "tot": {"L": 10}})
    v1 = Node(v1_b, {0: [11, 0]})
    v2_b = NodeBase(1, "facility", {"N": 15}, {"inc": {"L": 15},
                    "out": {"L": 15}, "tot": {"L": 15}})
    v2 = Node(v2_b, {1: [14, 0]})

    d1_b = NodeBase(2, "demand")
    d1 = Node(d1_b, {1: [0, -14]})
    d2_b = NodeBase(3, "demand")
    d2 = Node(d2_b, {0: [0, -10]})

    bin_b = NodeBase(4, "bin")
    bin_ = Node(bin_b)

    # Tariffs
    t_const_lvl1 = TariffLevel(1, 0, 0, 0)
    t_const_lvl2 = TariffLevel(0.9, 0, 2.7, 3)
    t_const_lvl3 = TariffLevel(0.8, 0, 4.8, 6)
    t_const_discount = Tariff(0, [t_const_lvl1, t_const_lvl2, t_const_lvl3], "W")

    t_lin_lvl1 = TariffLevel(1, 0.5, 0, 0)
    t_lin_lvl2 = TariffLevel(0.9, 0.5, 4.7, 3)
    t_lin_lvl3 = TariffLevel(0.8, 0.5, 8.0, 5)
    t_lin_discount = Tariff(1, [t_lin_lvl1, t_lin_lvl2, t_lin_lvl3], "W")

    t_const_lvl = TariffLevel(0, 0.01, 0, 0)
    t_const = Tariff(2, [t_const_lvl], "V")

    t_free_lvl = TariffLevel(0, 0, 0, 0)
    t_free = Tariff(3, [t_free_lvl], "V")

    # Instance
    ins = Instance(
                ["single_tariff", "no_perishable", "capacitated"], 
                {"time_periods": 2, 
                "c_types": ["N"], "c_types_n": 1,
                "c_properties": ["V", "W"], "c_properties_n": 2,
                "n_nodes_base": 5, "weight_cost": 1, "weight_green": 0,
                "transport_modes": ["L"], "storage_mode": ["C"],
                "co2_costs_per_g": 0.0001}
    )
    ins.commodities.extend([k0, k1])
    ins.tariffs.extend([t_const_discount, t_lin_discount, t_const, t_free])
    ins.nodes.extend([v1, v2, d1, d2, bin_])

    ## Arcs
    iso_cont_cap = {"N":{"V": 2, "W": 2}}
    str_cont_cap = {"N":{"V": 1, "W": 1}}
    uncapacitated = {"N": -1}

    # Transport
    a1_b = ArcBase(0, v1_b, v2_b, "L", 1, 1, t_const_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.04)
    a2_b = ArcBase(1, v2_b, v1_b, "L", 1, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.04)
    a1 = Arc(0, a1_b, 0, 1)
    a2 = Arc(1, a2_b, 0, 1)

    # Storage
    str1_b = ArcBase(2, v1_b, v1_b, "C", 1, 1, t_const, str_cont_cap, {"N": 20},
                        emissions_fkm=[1])
    str2_b = ArcBase(3, v2_b, v2_b, "C", 1, 1, t_const, str_cont_cap, {"N": 20},
                        emissions_fkm=[1])
    str1 = Arc(2, str1_b, 0, 1)
    str2 = Arc(3, str2_b, 0, 1)
    
    ins.arcs.extend([a1, a2, str1, str2])

    # Regional Demand
    da1_b = ArcBase(4, v1_b, d1_b, "O", 0, 0, t_free, str_cont_cap, uncapacitated)
    da2_b = ArcBase(5, v2_b, d2_b, "O", 0, 0, t_free, str_cont_cap, uncapacitated)
    idx = 3
    for t in [0, 1]:
        ins.arcs.append(Arc(idx+1, da1_b, t, t))
        ins.arcs.append(Arc(idx+2, da2_b, t, t))
        idx += 2
    
    # Other Demand
    dda1_b = ArcBase(6, v1_b, d2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.02)
    dda2_b = ArcBase(7, v2_b, d1_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.02)
    dda1 = Arc(idx+1, dda1_b, 0, 1)
    dda2 = Arc(idx+2, dda2_b, 0, 1)
    idx = idx + 2
    ins.arcs.extend([dda1, dda2])

    # Bin 
    facilities = [v1_b, v2_b]
    T = [0, 1]
    idx_b = 7
    for v in facilities:
        arc_base = ArcBase(idx_b + 1, v, bin_b, "O", 0, 0, t_free, str_cont_cap, 
                            uncapacitated)
        for t in T:
            idx += 1
            ins.arcs.append(Arc(idx, arc_base, t, t))
        idx_b += 1
    
    # Statistics
    ins.other["n_arcs"] = idx + 1

    return ins
    

def _gen_manual_instance_3() -> Instance:
    """Generate manual instance three. 
    
    Two modes, two product types, four products. No perishable products.
    T=3. Tests correct multi-type and multi-modality behaviour. """
    
    # Non-Perishable from 1/3 to 2
    k0 = Commodity(0, {"V": 2, "W": 2}, "N")
    # Non-Perishable from 2 to 1/3
    k1 = Commodity(1, {"V": 2, "W": 2}, "N")
    # Perishable from 1/3 to 2 
    k2 = Commodity(2, {"V": 2, "W": 2}, "F")
    # Perishable from 2 to 1/3
    k3 = Commodity(3, {"V": 2, "W": 2}, "F")

    # Nodes
    unlimited_storage = {"N": -1, "F": -1}
    unlimited_handling = {"inc": {"L": -1, "R": -1}, "out": {"L": -1, "R": -1},
                          "tot": {"L": -1, "R": -1}}
    v1_b = NodeBase(0, "facility", unlimited_storage, {"inc": {"L": -1},
                    "out": {"L": 8}, "tot": {"L": -1}})
    v1 = Node(v1_b, {0: [10, 10, 0], 2: [0, 5, 0]})
    v2_b = NodeBase(1, "facility", unlimited_storage, unlimited_handling)
    v2 = Node(v2_b, {1: [40, 20, 0], 3: [20, 20, 0]})
    v3_b = NodeBase(2, "facility", unlimited_storage, unlimited_handling)
    v3 = Node(v3_b, {0: [20, 0, 0], 2: [0, 5, 0]})

    # Region v1 and v3
    d1_b = NodeBase(3, "demand")
    d1 = Node(d1_b, {1: [0, -20, -20], 3: [0, -10, -10]})
    # Region v2
    d2_b = NodeBase(4, "demand")
    d2 = Node(d2_b, {0: [0, 0, -40], 2: [0, 0, -5]})

    bin_b = NodeBase(5, "bin")
    bin_ = Node(bin_b)

    # Tariffs
    t_const_lvl1 = TariffLevel(0.5, 0, 0, 0)
    t_const_lvl2 = TariffLevel(0.35, 0, 1.75, 5)
    t_const_discount = Tariff(0, [t_const_lvl1, t_const_lvl2], "W")

    t_lin_lvl1 = TariffLevel(1, 0.1, 0, 0)
    t_lin_lvl2 = TariffLevel(0.7, 0.07, 10.5, 8)
    t_lin_discount = Tariff(1, [t_lin_lvl1, t_lin_lvl2], "W")

    t_storage_lvl = TariffLevel(0, 0.1, 0, 0)
    t_storage = Tariff(2, [t_storage_lvl], "V")

    t_free_lvl = TariffLevel(0, 0, 0, 0)
    t_free = Tariff(3, [t_free_lvl], "V")

    # Instance
    ins = Instance(
                ["single_tariff", "capacitated"], 
                {"time_periods": 3, 
                "c_types": ["N", "F"], "c_types_n": 2,
                "c_properties": ["V", "W"], "c_properties_n": 2,
                "n_nodes_base": 6, "weight_cost": 1, "weight_green": 0,
                "transport_modes": ["L", "R"], "storage_mode": ["C"],
                "co2_costs_per_g": 0.0001}
    )
    ins.commodities.extend([k0, k1, k2, k3])
    ins.tariffs.extend([t_const_discount, t_lin_discount, t_storage, t_free])
    ins.nodes.extend([v1, v2, v3, d1, d2, bin_])

    ## Arcs
    iso_cont_cap = {"N":{"V": 10, "W": 10}, "F":{"V":10, "W":10}}
    str_cont_cap = {"N":{"V": 1, "W": 1}, "F":{"V":1, "W":1}}
    uncapacitated = {"N": -1, "F": -1}

    # Transport
    emissions_l = {"emissions_ykm": [10, 10], "emissions_fkm": [1, 1]}
    emissions_s = {"emissions_ykm": [0, 0], "emissions_fkm": [1, 1]}
    emissions_r = {"emissions_ykm": [5, 5], "emissions_fkm": [0, 0]}

    # Tariffs Equal for each product type
    t_lin_discount = [t_lin_discount] * 2
    t_const_discount = [t_const_discount] * 2
    t_storage = [t_storage] * 2
    t_free = [t_free] * 2

    a_l = []
    # Lorry
    a12_b = ArcBase(0, v1_b, v2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.02)
    a13_b = ArcBase(1, v1_b, v3_b, "L", 1, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03)
    a21_b = ArcBase(2, v2_b, v1_b, "L", 2, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.02)
    a23_b = ArcBase(3, v2_b, v3_b, "L", 2, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03 )
    a31_b = ArcBase(4, v3_b, v1_b, "L", 1, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03 )
    a32_b = ArcBase(5, v3_b, v2_b, "L", 2, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03 )
    a_l.extend([a12_b, a13_b, a21_b, a23_b, a31_b, a32_b])
    # Rail
    a23r_b = ArcBase(6, v2_b, v3_b, "R", 2, 2, t_const_discount, iso_cont_cap,
                    uncapacitated, **emissions_r, emissions_yh = 4, cost_yh = 0.04)
    a_l.extend([a23r_b])
    # Storage
    str1_b = ArcBase(7, v1_b, v1_b, "C", 1, 1, t_storage, str_cont_cap,
                    uncapacitated, **emissions_s)
    str2_b = ArcBase(8, v2_b, v2_b, "C", 1, 1, t_storage, str_cont_cap,
                    uncapacitated, **emissions_s)
    str3_b = ArcBase(9, v3_b, v3_b, "C", 1, 1, t_storage, str_cont_cap,
                    uncapacitated, **emissions_s)
    a_l.extend([str1_b, str2_b, str3_b])
    # Regional Demand
    da1_b = ArcBase(10, v1_b, d1_b, "O", 0, 0, t_free, str_cont_cap, 
                    uncapacitated, [0,0], [0,0])
    da2_b = ArcBase(11, v2_b, d2_b, "O", 0, 0, t_free, str_cont_cap, 
                    uncapacitated, [0,0], [0,0])
    da3_b = ArcBase(12, v3_b, d1_b, "O", 0, 0, t_free, str_cont_cap, 
                    uncapacitated, [0,0], [0,0])
    a_l.extend([da1_b, da2_b, da3_b])
    # Other Demand
    dda1_b = ArcBase(13, v1_b, d2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 1, cost_fh = 0.01 )
    dda2_b = ArcBase(14, v2_b, d1_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 1, cost_fh = 0.01 )
    dda3_b = ArcBase(15, v3_b, d2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 1, cost_fh = 0.02 )
    a_l.extend([dda1_b, dda2_b, dda3_b])

    # Time Expand Arcs and Add to Instance
    idx = 0
    for arc in a_l:
        for t in [0, 1, 2]:
            if t + arc.time <= 2:
                ins.arcs.append(Arc(idx, arc, t, t+arc.time))
                idx += 1

    # Bin 
    facilities = [v1_b, v2_b, v3_b]
    T = [0, 1, 2]
    idx_b = 15
    for v in facilities:
        arc_base = ArcBase(idx_b + 1, v, bin_b, "O", 0, 0, t_free, str_cont_cap,
                    uncapacitated, [0,0], [0,0])
        for t in T:
            ins.arcs.append(Arc(idx, arc_base, t, t))
            idx += 1
        idx_b += 1
    
    # Statistics
    ins.other["n_arcs"] = idx

    return ins


def _gen_manual_instance_4() -> Instance:
    """Generate manual instance four. 
    
    Two modes, one product types, four products. No perishable products.
    T=3. Tests correctly allocating multiple products to one tariff. """
    
    # Non-Perishable from 1/3 to 2
    k0 = Commodity(0, {"V": 2, "W": 2}, "N")
    # Non-Perishable from 2 to 1/3
    k1 = Commodity(1, {"V": 2, "W": 2}, "N")
    # Perishable from 1/3 to 2 
    k2 = Commodity(2, {"V": 2, "W": 2}, "N")
    # Perishable from 2 to 1/3
    k3 = Commodity(3, {"V": 2, "W": 2}, "N")

    # Nodes
    unlimited_storage = {"N": -1, "F": -1}
    unlimited_handling = {"inc": {"L": -1, "R": -1}, "out": {"L": -1, "R": -1},
                          "tot": {"L": -1, "R": -1}}
    v1_b = NodeBase(0, "facility", unlimited_storage, {"inc": {"L": -1},
                    "out": {"L": 8}, "tot": {"L": -1}})
    v1 = Node(v1_b, {0: [10, 10, 0], 2: [0, 5, 0]})
    v2_b = NodeBase(1, "facility", unlimited_storage, unlimited_handling)
    v2 = Node(v2_b, {1: [40, 20, 0], 3: [20, 20, 0]})
    v3_b = NodeBase(2, "facility", unlimited_storage, unlimited_handling)
    v3 = Node(v3_b, {0: [20, 0, 0], 2: [0, 5, 0]})

    # Region v1 and v3
    d1_b = NodeBase(3, "demand")
    d1 = Node(d1_b, {1: [0, -20, -20], 3: [0, -10, -10]})
    # Region v2
    d2_b = NodeBase(4, "demand")
    d2 = Node(d2_b, {0: [0, 0, -40], 2: [0, 0, -5]})

    bin_b = NodeBase(5, "bin")
    bin_ = Node(bin_b)

    # Tariffs
    t_const_lvl1 = TariffLevel(0.5, 0, 0, 0)
    t_const_lvl2 = TariffLevel(0.35, 0, 1.75, 5)
    t_const_discount = Tariff(0, [t_const_lvl1, t_const_lvl2], "W")

    t_lin_lvl1 = TariffLevel(1, 0.1, 0, 0)
    t_lin_lvl2 = TariffLevel(0.7, 0.07, 10.5, 8)
    t_lin_discount = Tariff(1, [t_lin_lvl1, t_lin_lvl2], "W")

    t_storage_lvl = TariffLevel(0, 0.1, 0, 0)
    t_storage = Tariff(2, [t_storage_lvl], "V")

    t_free_lvl = TariffLevel(0, 0, 0, 0)
    t_free = Tariff(3, [t_free_lvl], "V")

    # Instance
    ins = Instance(
                ["single_tariff", "capacitated"], 
                {"time_periods": 3, 
                "c_types": ["N"], "c_types_n": 1,
                "c_properties": ["V", "W"], "c_properties_n": 2,
                "n_nodes_base": 6, "weight_cost": 1, "weight_green": 0,
                "transport_modes": ["L", "R"], "storage_mode": ["C"],
                "co2_costs_per_g": 0.0001}
    )
    ins.commodities.extend([k0, k1, k2, k3])
    ins.tariffs.extend([t_const_discount, t_lin_discount, t_storage, t_free])
    ins.nodes.extend([v1, v2, v3, d1, d2, bin_])

    ## Arcs
    iso_cont_cap = {"N":{"V": 10, "W": 10}, "F":{"V":10, "W":10}}
    str_cont_cap = {"N":{"V": 1, "W": 1}, "F":{"V":1, "W":1}}
    uncapacitated = {"N": -1, "F": -1}

    # Transport
    emissions_l = {"emissions_ykm": [10], "emissions_fkm": [1]}
    emissions_s = {"emissions_ykm": [0], "emissions_fkm": [1]}
    emissions_r = {"emissions_ykm": [5], "emissions_fkm": [0]}

    a_l = []
    # Lorry
    a12_b = ArcBase(0, v1_b, v2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.02)
    a13_b = ArcBase(1, v1_b, v3_b, "L", 1, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03)
    a21_b = ArcBase(2, v2_b, v1_b, "L", 2, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.02)
    a23_b = ArcBase(3, v2_b, v3_b, "L", 2, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03 )
    a31_b = ArcBase(4, v3_b, v1_b, "L", 1, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03 )
    a32_b = ArcBase(5, v3_b, v2_b, "L", 2, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03 )
    a_l.extend([a12_b, a13_b, a21_b, a23_b, a31_b, a32_b])
    # Rail
    a23r_b = ArcBase(6, v2_b, v3_b, "R", 2, 2, t_const_discount, iso_cont_cap,
                    uncapacitated, **emissions_r, emissions_yh = 4, cost_yh = 0.04)
    a_l.extend([a23r_b])
    # Storage
    str1_b = ArcBase(7, v1_b, v1_b, "C", 1, 1, t_storage, str_cont_cap,
                    uncapacitated, **emissions_s)
    str2_b = ArcBase(8, v2_b, v2_b, "C", 1, 1, t_storage, str_cont_cap,
                    uncapacitated, **emissions_s)
    str3_b = ArcBase(9, v3_b, v3_b, "C", 1, 1, t_storage, str_cont_cap,
                    uncapacitated, **emissions_s)
    a_l.extend([str1_b, str2_b, str3_b])
    # Regional Demand
    da1_b = ArcBase(10, v1_b, d1_b, "O", 0, 0, t_free, str_cont_cap, 
                    uncapacitated, [0], [0])
    da2_b = ArcBase(11, v2_b, d2_b, "O", 0, 0, t_free, str_cont_cap, 
                    uncapacitated, [0], [0])
    da3_b = ArcBase(12, v3_b, d1_b, "O", 0, 0, t_free, str_cont_cap, 
                    uncapacitated, [0], [0])
    a_l.extend([da1_b, da2_b, da3_b])
    # Other Demand
    dda1_b = ArcBase(13, v1_b, d2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 1, cost_fh = 0.01 )
    dda2_b = ArcBase(14, v2_b, d1_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 1, cost_fh = 0.01 )
    dda3_b = ArcBase(15, v3_b, d2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 1, cost_fh = 0.02 )
    a_l.extend([dda1_b, dda2_b, dda3_b])

    # Time Expand Arcs and Add to Instance
    idx = 0
    for arc in a_l:
        for t in [0, 1, 2]:
            if t + arc.time <= 2:
                ins.arcs.append(Arc(idx, arc, t, t+arc.time))
                idx += 1

    # Bin 
    facilities = [v1_b, v2_b, v3_b]
    T = [0, 1, 2]
    idx_b = 15
    for v in facilities:
        arc_base = ArcBase(idx_b + 1, v, bin_b, "O", 0, 0, t_free, str_cont_cap,
                    uncapacitated, [0], [0])
        for t in T:
            ins.arcs.append(Arc(idx, arc_base, t, t))
            idx += 1
        idx_b += 1
    
    # Statistics
    ins.other["n_arcs"] = idx

    return ins


def _gen_manual_instance_5() -> Instance:
    """Generate manual instance three. 
    
    Two modes, two product types, four products. Two perishable products.
    T=3. Tests correct multi-type and multi-modality behaviour. """
    
    # Non-Perishable from 1/3 to 2
    k0 = Commodity(0, {"V": 2, "W": 2}, "N")
    # Non-Perishable from 2 to 1/3
    k1 = Commodity(1, {"V": 2, "W": 2}, "N")
    # Perishable from 1/3 to 2 
    k2 = Commodity(2, {"V": 2, "W": 2}, "F", lifetime=1)
    # Perishable from 2 to 1/3
    k3 = Commodity(3, {"V": 2, "W": 2}, "F", lifetime=1)

    # Nodes
    unlimited_storage = {"N": -1, "F": -1}
    unlimited_handling = {"inc": {"L": -1, "R": -1}, "out": {"L": -1, "R": -1},
                          "tot": {"L": -1, "R": -1}}
    v1_b = NodeBase(0, "facility", unlimited_storage, {"inc": {"L": -1},
                    "out": {"L": 8}, "tot": {"L": -1}})
    v1 = Node(v1_b, {0: [10, 10, 0], 2: [0, 5, 0]})
    v2_b = NodeBase(1, "facility", unlimited_storage, unlimited_handling)
    v2 = Node(v2_b, {1: [40, 20, 0], 3: [20, 20, 0]})
    v3_b = NodeBase(2, "facility", unlimited_storage, unlimited_handling)
    v3 = Node(v3_b, {0: [20, 0, 0], 2: [0, 5, 0]})

    # Region v1 and v3
    d1_b = NodeBase(3, "demand")
    d1 = Node(d1_b, {1: [0, -20, -20], 3: [0, -10, -10]})
    # Region v2
    d2_b = NodeBase(4, "demand")
    d2 = Node(d2_b, {0: [0, 0, -40], 2: [0, 0, -5]})

    bin_b = NodeBase(5, "bin")
    bin_ = Node(bin_b)

    # Tariffs
    t_const_lvl1 = TariffLevel(0.5, 0, 0, 0)
    t_const_lvl2 = TariffLevel(0.35, 0, 1.75, 5)
    t_const_discount = Tariff(0, [t_const_lvl1, t_const_lvl2], "W")

    t_lin_lvl1 = TariffLevel(1, 0.1, 0, 0)
    t_lin_lvl2 = TariffLevel(0.7, 0.07, 10.5, 8)
    t_lin_discount = Tariff(1, [t_lin_lvl1, t_lin_lvl2], "W")

    t_storage_lvl = TariffLevel(0, 0.1, 0, 0)
    t_storage = Tariff(2, [t_storage_lvl], "V")

    t_free_lvl = TariffLevel(0, 0, 0, 0)
    t_free = Tariff(3, [t_free_lvl], "V")

    # Instance
    ins = Instance(
                ["single_tariff", "multi_source", "capacitated"], 
                {"time_periods": 3, 
                "c_types": ["N", "F"], "c_types_n": 2,
                "c_properties": ["V", "W"], "c_properties_n": 2,
                "n_nodes_base": 6, "weight_cost": 1, "weight_green": 0,
                "transport_modes": ["L", "R"], "storage_mode": ["C"],
                "co2_costs_per_g": 0.0001}
    )
    ins.commodities.extend([k0, k1, k2, k3])
    ins.tariffs.extend([t_const_discount, t_lin_discount, t_storage, t_free])
    ins.nodes.extend([v1, v2, v3, d1, d2, bin_])

    ## Arcs
    iso_cont_cap = {"N":{"V": 10, "W": 10}, "F":{"V":10, "W":10}}
    str_cont_cap = {"N":{"V": 1, "W": 1}, "F":{"V":1, "W":1}}
    uncapacitated = {"N": -1, "F": -1}

    # Transport
    emissions_l = {"emissions_ykm": [10, 10], "emissions_fkm": [1, 1]}
    emissions_s = {"emissions_ykm": [0, 0], "emissions_fkm": [1, 1]}
    emissions_r = {"emissions_ykm": [5, 5], "emissions_fkm": [0, 0]}

    # Tariffs Equal for each product type
    t_lin_discount = [t_lin_discount] * 2
    t_const_discount = [t_const_discount] * 2
    t_storage = [t_storage] * 2
    t_free = [t_free] * 2

    a_l = []
    # Lorry
    a12_b = ArcBase(0, v1_b, v2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.02)
    a13_b = ArcBase(1, v1_b, v3_b, "L", 1, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03)
    a21_b = ArcBase(2, v2_b, v1_b, "L", 2, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.02)
    a23_b = ArcBase(3, v2_b, v3_b, "L", 2, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03 )
    a31_b = ArcBase(4, v3_b, v1_b, "L", 1, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03 )
    a32_b = ArcBase(5, v3_b, v2_b, "L", 2, 1, t_lin_discount, iso_cont_cap,
                    uncapacitated, **emissions_l, emissions_fh = 2, cost_fh = 0.03 )
    a_l.extend([a12_b, a13_b, a21_b, a23_b, a31_b, a32_b])
    # Rail
    a23r_b = ArcBase(6, v2_b, v3_b, "R", 2, 2, t_const_discount, iso_cont_cap,
                    uncapacitated, **emissions_r, emissions_yh = 4, cost_yh = 0.04)
    a_l.extend([a23r_b])
    # Storage
    str1_b = ArcBase(7, v1_b, v1_b, "C", 1, 1, t_storage, str_cont_cap,
                    uncapacitated, **emissions_s)
    str2_b = ArcBase(8, v2_b, v2_b, "C", 1, 1, t_storage, str_cont_cap,
                    uncapacitated, **emissions_s)
    str3_b = ArcBase(9, v3_b, v3_b, "C", 1, 1, t_storage, str_cont_cap,
                    uncapacitated, **emissions_s)
    a_l.extend([str1_b, str2_b, str3_b])
    # Regional Demand
    da1_b = ArcBase(10, v1_b, d1_b, "O", 0, 0, t_free, str_cont_cap, 
                    uncapacitated, [0,0], [0,0])
    da2_b = ArcBase(11, v2_b, d2_b, "O", 0, 0, t_free, str_cont_cap, 
                    uncapacitated, [0,0], [0,0])
    da3_b = ArcBase(12, v3_b, d1_b, "O", 0, 0, t_free, str_cont_cap, 
                    uncapacitated, [0,0], [0,0])
    a_l.extend([da1_b, da2_b, da3_b])
    # Other Demand
    dda1_b = ArcBase(13, v1_b, d2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 1, cost_fh = 0.01 )
    dda2_b = ArcBase(14, v2_b, d1_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 1, cost_fh = 0.01 )
    dda3_b = ArcBase(15, v3_b, d2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, **emissions_l, emissions_fh = 1, cost_fh = 0.02 )
    a_l.extend([dda1_b, dda2_b, dda3_b])

    # Time Expand Arcs and Add to Instance
    idx = 0
    for arc in a_l:
        for t in [0, 1, 2]:
            if t + arc.time <= 2:
                ins.arcs.append(Arc(idx, arc, t, t+arc.time))
                idx += 1

    # Bin 
    facilities = [v1_b, v2_b, v3_b]
    T = [0, 1, 2]
    idx_b = 15
    for v in facilities:
        arc_base = ArcBase(idx_b + 1, v, bin_b, "O", 0, 0, t_free, str_cont_cap,
                    uncapacitated, [0,0], [0,0])
        for t in T:
            ins.arcs.append(Arc(idx, arc_base, t, t))
            idx += 1
        idx_b += 1
    
    # Statistics
    ins.other["n_arcs"] = idx

    return ins


def _gen_manual_instance_6() -> Instance:
    """Generate manual instance two. Single mode, two products, handling capacities. """

    k0 = Commodity(0, {"V": 1, "W": 1}, "N")
    k1 = Commodity(1, {"V": 1, "W": 1}, "N")

    # Nodes
    v1_b = NodeBase(0, "facility", {"N": 10}, {"inc": {"L": -1},
                    "out": {"L": -1}, "tot": {"L": -1}})
    v1 = Node(v1_b, {0: [11, 0]})
    v2_b = NodeBase(1, "facility", {"N": 15}, {"inc": {"L": -1},
                    "out": {"L": -1}, "tot": {"L": -1}})
    v2 = Node(v2_b, {1: [14, 0]})

    d1_b = NodeBase(2, "demand")
    d1 = Node(d1_b, {1: [0, -14]})
    d2_b = NodeBase(3, "demand")
    d2 = Node(d2_b, {0: [0, -10]})

    bin_b = NodeBase(4, "bin")
    bin_ = Node(bin_b)

    # Tariffs
    t_const_lvl1 = TariffLevel(1, 0, 0, 0)
    t_const_lvl2 = TariffLevel(0.9, 0, 2.7, 3)
    t_const_lvl3 = TariffLevel(0.8, 0, 4.8, 6)
    t_const_discount = Tariff(0, [t_const_lvl1, t_const_lvl2, t_const_lvl3], "W")

    t_lin_lvl1 = TariffLevel(1, 0.5, 0, 0)
    t_lin_lvl2 = TariffLevel(0.9, 0.5, 4.7, 3)
    t_lin_lvl3 = TariffLevel(0.8, 0.5, 8.0, 5)
    t_lin_discount = Tariff(1, [t_lin_lvl1, t_lin_lvl2, t_lin_lvl3], "W")

    t_const_lvl = TariffLevel(0, 0.01, 0, 0)
    t_const = Tariff(2, [t_const_lvl], "V")

    t_free_lvl = TariffLevel(0, 0, 0, 0)
    t_free = Tariff(3, [t_free_lvl], "V")

    # Instance
    ins = Instance(
                ["single_tariff", "no_perishable"], 
                {"time_periods": 2, 
                "c_types": ["N"], "c_types_n": 1,
                "c_properties": ["V", "W"], "c_properties_n": 2,
                "n_nodes_base": 5, "weight_cost": 1, "weight_green": 0,
                "transport_modes": ["L"], "storage_mode": ["C"],
                "co2_costs_per_g": 0.0001}
    )
    ins.commodities.extend([k0, k1])
    ins.tariffs.extend([t_const_discount, t_lin_discount, t_const, t_free])
    ins.nodes.extend([v1, v2, d1, d2, bin_])

    ## Arcs
    iso_cont_cap = {"N":{"V": 2, "W": 2}}
    str_cont_cap = {"N":{"V": 1, "W": 1}}
    uncapacitated = {"N": -1}

    # Transport
    a1_b = ArcBase(0, v1_b, v2_b, "L", 1, 1, t_const_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.04)
    a2_b = ArcBase(1, v2_b, v1_b, "L", 1, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.04)
    a1 = Arc(0, a1_b, 0, 1)
    a2 = Arc(1, a2_b, 0, 1)

    # Storage
    str1_b = ArcBase(2, v1_b, v1_b, "C", 1, 1, t_const, str_cont_cap, {"N": 20},
                        emissions_fkm=[1])
    str2_b = ArcBase(3, v2_b, v2_b, "C", 1, 1, t_const, str_cont_cap, {"N": 20},
                        emissions_fkm=[1])
    str1 = Arc(2, str1_b, 0, 1)
    str2 = Arc(3, str2_b, 0, 1)
    
    ins.arcs.extend([a1, a2, str1, str2])

    # Regional Demand
    da1_b = ArcBase(4, v1_b, d1_b, "O", 0, 0, t_free, str_cont_cap, uncapacitated)
    da2_b = ArcBase(5, v2_b, d2_b, "O", 0, 0, t_free, str_cont_cap, uncapacitated)
    idx = 3
    for t in [0, 1]:
        ins.arcs.append(Arc(idx+1, da1_b, t, t))
        ins.arcs.append(Arc(idx+2, da2_b, t, t))
        idx += 2
    
    # Other Demand
    dda1_b = ArcBase(6, v1_b, d2_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.02)
    dda2_b = ArcBase(7, v2_b, d1_b, "L", 2, 1, t_lin_discount, iso_cont_cap, 
                    uncapacitated, [1], [1], 1, 1, 0, 0.02)
    dda1 = Arc(idx+1, dda1_b, 0, 1)
    dda2 = Arc(idx+2, dda2_b, 0, 1)
    idx = idx + 2
    ins.arcs.extend([dda1, dda2])

    # Bin 
    facilities = [v1_b, v2_b]
    T = [0, 1]
    idx_b = 7
    for v in facilities:
        arc_base = ArcBase(idx_b + 1, v, bin_b, "O", 0, 0, t_free, str_cont_cap, 
                            uncapacitated)
        for t in T:
            idx += 1
            ins.arcs.append(Arc(idx, arc_base, t, t))
        idx_b += 1
    
    # Statistics
    ins.other["n_arcs"] = idx + 1

    return ins