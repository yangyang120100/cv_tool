THING_CLASSES = ['traffic_cone', 'street_light', 'road_block', 'car', 'bicycle', 'person', 'bus', 'traffic_light',
                 'motorcycle', 'boom_barrier', 'tree_trunk', 'rider', 'animal', 'truck', 'on_rails', 'caravan',
                 'trailer', 'rock', 'pole', 'traffic_sign', 'misc_sign', 'kick_scooter', 'heavy_machinery',
                 'container', 'barrel', 'military_vehicle']
STUFF_CLASSES = ['undefined', 'snow', 'cobble', 'obstacle', 'leaves', 'bikeway', 'ego_vehicle',
                 'pedestrian_crossing', 'road_marking', 'forest', 'bush', 'moss', 'sidewalk', 'curb', 'asphalt',
                 'gravel', 'rail_track', 'tree_crown', 'debris', 'crops', 'soil', 'building', 'wall', 'fence',
                 'guard_rail', 'bridge', 'tunnel', 'barrier_tape', 'low_grass', 'high_grass', 'scenery_vegetation',
                 'sky', 'water', 'wire', 'outlier', 'hedge', 'pipe', 'tree_root']
THING_COLORS = [(0, 255, 255), (166, 111, 0), (166, 0, 0), (98, 118, 183), (67, 77, 0), (255, 176, 143),
                (135, 125, 153), (0, 68, 27), (1, 198, 79), (45, 54, 52), (146, 170, 255), (255, 239, 221),
                (53, 0, 0), (75, 79, 123), (153, 194, 161), (24, 0, 48), (216, 166, 10), (1, 33, 55),
                (178, 185, 192), (153, 255, 194), (9, 30, 0), (98, 0, 111), (159, 208, 250), (154, 138, 255),
                (0, 208, 208), (64, 64, 64)]
STUFF_COLORS = [(0, 0, 0), (160, 87, 209), (255, 52, 255), (70, 74, 255), (65, 137, 0), (89, 0, 163),
                (229, 219, 255), (0, 73, 122), (172, 255, 99), (7, 0, 90), (147, 150, 128), (189, 168, 180),
                (255, 93, 59), (83, 59, 74), (128, 47, 255), (90, 97, 97), (0, 121, 107), (160, 194, 0),
                (76, 111, 136), (237, 134, 0), (0, 97, 209), (73, 51, 1), (111, 132, 0), (0, 181, 255),
                (237, 255, 194), (191, 121, 160), (68, 7, 204), (89, 196, 190), (102, 189, 12), (255, 195, 238),
                (117, 109, 69), (104, 123, 183), (161, 135, 122), (0, 140, 255), (102, 141, 120), (23, 211, 232),
                (0, 0, 221), (132, 164, 196)]
THING_CLASSES_ID = [1, 6, 10, 12, 13, 14, 15, 19, 20, 25, 28, 32, 33, 34, 35, 36, 37, 40, 45, 46, 47, 49, 57, 58,
                    60, 63]
STUFF_CLASSES_ID = [0, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 21, 22, 23, 24, 26, 27, 29, 30, 31, 38, 39, 41, 42, 43,
                    44, 48, 50, 51, 52, 53, 54, 55, 56, 59, 61, 62]

thing_to_contiguous_id_map = {
    THING_CLASSES_ID[i]: i
    for i in range(len(THING_CLASSES))
}

stuff_to_contiguous_id_map = {
    STUFF_CLASSES_ID[i]: i
    for i in range(len(STUFF_CLASSES))
}

metadata = {
    "thing_classes": THING_CLASSES,
    "stuff_classes": STUFF_CLASSES,
    "thing_colors": THING_COLORS,
    "stuff_colors": STUFF_COLORS,
    "thing_dataset_id_to_contiguous_id": thing_to_contiguous_id_map,
    "stuff_dataset_id_to_contiguous_id": stuff_to_contiguous_id_map,
}

# --- 构建映射 ---
# Thing 映射：原始 ID -> 连续 ID (0~25)
thing_dataset_id_to_contiguous_id = {orig_id: idx for idx, orig_id in enumerate(THING_CLASSES_ID)}
# Stuff 映射：原始 ID -> 连续 ID (0~37)
stuff_dataset_id_to_contiguous_id = {orig_id: idx for idx, orig_id in enumerate(STUFF_CLASSES_ID)}

# 合并后的所有类别名称（thing + stuff）
all_classes = THING_CLASSES + STUFF_CLASSES
# 合并后的所有颜色
all_colors = THING_COLORS + STUFF_COLORS

all_map=thing_dataset_id_to_contiguous_id|stuff_dataset_id_to_contiguous_id

print('')