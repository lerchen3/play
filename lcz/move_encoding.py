"""Mapping between UCI strings and action indices used by the network."""

from collections import OrderedDict

FILES = "abcdefgh"
RANKS = "12345678"
SQUARE_NAMES = [f + r for f in FILES for r in RANKS]

MOVE_INDEX = OrderedDict()
index = 0
for from_sq in SQUARE_NAMES:
    for to_sq in SQUARE_NAMES:
        uci = from_sq + to_sq
        MOVE_INDEX[uci] = index
        index += 1
        
        # Add promotions for ALL moves (even impossible ones) for consistency with existing models
        for promo in ['q', 'r', 'b', 'n']:
            uci_p = uci + promo
            MOVE_INDEX[uci_p] = index
            index += 1

INDEX_MOVE = {v: k for k, v in MOVE_INDEX.items()} 
