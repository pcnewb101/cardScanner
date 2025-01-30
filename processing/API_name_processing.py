from UPDATES_ONLY.retrieve_all_names import all_card_names
import sys
import os
import numpy as np
import re

# Get the absolute path of the project root (one level up from 'processing')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to sys.path so Python can find UPDATES_ONLY
sys.path.append(project_root)


#Removes duplicate values
all_unique_names = np.unique(all_card_names)
# Debugging statement
# print(f"Unique info: Len: {len(all_unique_names)}, Sample: {all_unique_names[:10]}")


pokemon_names = []


#Removes non-english or garbage input
for name in all_unique_names:
    if isinstance(name, bytes):  # Check if it's bytes and decode it
        print(f"Name: {name}, decoded with utf-8")
        text = name.decode("utf-8")
        pokemon_names.append(re.sub(r"[^a-zA-Z0-9\s/,\"'_-]", "", text))
    else:
        pokemon_names.append(re.sub(r"[^a-zA-Z0-9\s/,\"'_-]", "", name))

#checks for duplicates/repeat empty names and reduces to one entry
pokemon_names = np.unique(pokemon_names)

