import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import get_negative, get_positive

games = ["platform", "slide", "vertical", "cave", "cave_portal", "cave_doors", "crates"]
neg_min = 100000
pos_min = 100000
for game in games:
    X_pos, _= get_positive(game)
    X_neg, _ = get_negative(game)
    if len(X_pos) < pos_min:
        pos_min = len(X_pos)
    if len(X_neg) < neg_min:
        neg_min = len(X_neg)
    print("game: ", game)
    print("pos: ", len(X_pos))
    print("neg: ", len(X_neg))
    print("sum: ", len(X_pos) + len(X_neg))
    print("------------------------------")


print("** Negative min: ", neg_min)
print("** Positive min: ", pos_min)

# game:  platform
# pos:  27000
# neg:  17000
# sum:  44000
# ------------------------------
# game:  slide
# pos:  18000
# neg:  17000
# sum:  35000
# ------------------------------
# game:  vertical
# pos:  11999
# neg:  60000
# sum:  71999
# ------------------------------
# game:  cave
# pos:  4579
# neg:  24000
# sum:  28579
# ------------------------------
# game:  cave_portal
# pos:  38000
# neg:  47000
# sum:  85000
# ------------------------------
# game:  cave_doors
# pos:  44000
# neg:  22000
# sum:  66000
# ------------------------------
# game:  crates
# pos:  13000
# neg:  1000
# sum:  14000
# ------------------------------
# ** Negative min:  1000
# ** Positive min:  4579