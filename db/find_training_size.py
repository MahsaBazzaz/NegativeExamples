import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import get_positive_db, get_negative_db

game = "cave_treasures"

X_pos_1= get_positive_db(game, 1)
X_pos_2= get_positive_db(game, 2)
X_pos_3= get_positive_db(game, 3)
X_neg_1= get_negative_db(game, 1)
X_neg_2= get_negative_db(game, 2)
X_neg_3= get_negative_db(game, 1)


print("game: ", game)
print("1+: ", len(X_pos_1))
print("2+: ", len(X_pos_2))
print("3+: ", len(X_pos_3))

print("1-: ", len(X_neg_1))
print("2-: ", len(X_neg_2))
print("3-: ", len(X_neg_2))

print("sum+: ", len(X_pos_1) + len(X_pos_2) + len(X_pos_3))
print("sum-: ", len(X_neg_1) + len(X_neg_2) + len(X_neg_2))

print("------------------------------")

# game:  cave_treasures
# 1+:  2900
# 2+:  2900
# 3+:  2900
# 1-:  2900
# 2-:  2900
# 3-:  2900
# sum+:  8700
# sum-:  8700
# ------------------------------
# ------------------------------
# game:  mario
# 1+:  2899
# 2+:  2899
# 3+:  2900
# 1-:  2900
# 2-:  2900
# 3-:  2900
# sum+:  8698
# sum-:  8700
# ------------------------------
