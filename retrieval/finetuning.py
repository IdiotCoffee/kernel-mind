# -----------------------------------
# Depth penalty
# -----------------------------------

DEPTH_PENALTY = 0.15


# -----------------------------------
# Node-type weights
# -----------------------------------

TYPE_WEIGHTS = {
    "function": 1.0,
    "method": 1.2,
    "class": 0.6,
    "module": 0.5,
}


# -----------------------------------
# Connectivity boost
# -----------------------------------

CONNECTIVITY_WEIGHT = 0.05


# -----------------------------------
# Query overlap
# -----------------------------------

QUERY_MATCH_WEIGHT = 0.8

PROPAGATION_WEIGHT = 2.0

PROXIMITY_DEPTH_0_BOOST = 1.0

PROXIMITY_DEPTH_1_BOOST = 0.5
