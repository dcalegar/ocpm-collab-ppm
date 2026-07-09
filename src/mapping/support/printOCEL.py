from pm4py.read import read_ocel2_json
from pm4py.ocel import (
    ocel_get_object_types,
    ocel_object_type_activities,
    discover_oc_petri_net,
    discover_ocdfg
)
from pm4py.vis import view_ocpn, save_vis_ocpn, save_vis_ocdfg

modelname = "collectivelog_real4_collab"

# 1. Load OCEL 2.0 JSON
ocel = read_ocel2_json(modelname + ".jsonocel")

# 2. Verify log content
print("Object types:")
print(ocel_get_object_types(ocel))

print("\nActivities per object type:")
print(ocel_object_type_activities(ocel))

# 3. Discover Object-Centric Petri Net
ocpn = discover_oc_petri_net(ocel, inductive_miner_variant="imd")

# 4. Save OC-PN image
save_vis_ocpn(ocpn, f"{modelname}_ocpn.png", format="png", bgcolor="white")

# 5. Discover OC-DFG
ocdfg = discover_ocdfg(ocel)

# 6. Save OC-DFG image
save_vis_ocdfg(ocdfg, f"{modelname}_ocdfg.png", annotation="frequency", bgcolor="white")