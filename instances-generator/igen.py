"""
Author: Lukas Gosch
Date: 17.12.2020
Description:
    Generate manual or random instances for the generalized tactical
    transportation problem. 
Usage:
    Examples:
        - Manual instance generation: 
            python igen.py -m 5 -o ../data/m05.json
        - Display help:
            python igen.py -h
"""

import argparse
import sys

from gen_manual import gen_manual_instance
from gen_random import gen_random_instance


def parse(argv):
    """ Parse command-line options. """
    parser = argparse.ArgumentParser(description="Instance Generator")
    # General Arguments 
    parser.add_argument("-o", "--out-file", type=str,
            help="Output file name or path.")
    parser.add_argument("-i", "--indent", type=int, # overwriteable in config
            help="Output JSON indent level, None if not set.")
    parser.add_argument("-j", "--only-json", action="store_true",
            help="Directly store arcs in JSON. Can result in a very large file!"
                +"Default False. ")
    # Manual Instance
    parser.add_argument("-m", "--manual", type=int,
            help="Generate manual instance with given id.")
    # Big (random) Instance Generation
    parser.add_argument("-C", "--companies", type=int,
            help="Number of companies (groups) commodities are grouped into.")
    parser.add_argument("-R", "--regions", type=int, nargs=3,
            help="List of demand regions split. Example: -R 5 10 0 means " + 
                 "take 5 large regions, 10 medium and 0 small ones.")
    parser.add_argument("--use-regions", nargs='*',
            help="Optional. Give full list of region names to use. If set " + 
                 "overwrites -R setting. Example: --use-regions Wien Budapest")
    parser.add_argument("-W", "--warehouses", type=int, nargs=3,
            help="Warehouse split. Example: -W 4 2 1 means 4 warehouses per " +
                 "large region, 2 per medium and 1 per small.", default=[4, 2, 1])
    parser.add_argument("-K", "--commodities", type=int,
            help="Number of total commodities.")
    parser.add_argument("-T", "--time-periods", type=int,
            help="Time horizont.")
    parser.add_argument("-F", "--fresh-commodities", type=int, default=0,
            help="Number of pershable (fresh) commodities.")
    parser.add_argument("-M", "--modes", nargs='*',
            help="Optional. Give full list of modes to use. At least 'L' "
                +"necessary. Can use 'R' and 'S'. Default: ['L', 'R', 'S']")
    parser.add_argument("--capacitate", required=False,
            choices=["tight", "T", "loose", "L"],
            help="If instance should be tightly or loosly capacitated.")
    parser.add_argument("--max-demand", type=int, default=20,
            help="Maximal demand per commody in tons.")
    parser.add_argument("--direct-delivery", required=False, 
            action='store_true',
            help="If instance should represent direct delivery only.")
    parser.add_argument("-s", "--seed", type=int,
            help="Seed for random instances.")
    parser.add_argument("-d", "--distance-matrix", 
                        default="/home/gosch/proj/pi/igen/"
                               +"combine_multimodal_distances/distances.txt")
    parser.add_argument("-t", "--traveltime-matrix",
                        default="/home/gosch/proj/pi/igen/"
                               +"combine_multimodal_distances/traveltimes.txt")
    parser.add_argument("-l", "--locations",
                        default="/home/gosch/proj/pi/igen/"
                               +"combine_multimodal_distances/locations.txt")
    return parser.parse_args(argv[1:])


def main(argv):
    args = parse(argv)

    # Generate instance
    if args.manual is not None:
        ins = gen_manual_instance(args.manual)
    else:
        ins = gen_random_instance(args.companies, args.regions, args.use_regions,
                                  args.warehouses, args.commodities, args.time_periods, 
                                  args.fresh_commodities, args.modes,
                                  args.capacitate, args.max_demand,
                                  args.direct_delivery,
                                  args.seed, args.distance_matrix, 
                                  args.traveltime_matrix, args.locations)

    # Write generated instance to disk
    if args.out_file is None:
        args.out_file = "out.txt"
    ins.to_disk(args.out_file, indent=args.indent, only_json=args.only_json)
    
    return 0


if __name__ == "__main__":
    main(sys.argv)
