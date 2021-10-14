import pickle as pkl
from inkml import Inkml, Segment
import random
from tqdm import tqdm
import numpy as np
from config import CONFIG
import os

# ------------------- CONFIGURATION -------------------------------
FILTER = True
FILTER_STR = "Train_2014"
# first step: decomposition. Set False if already run the code
SUB_SPLITER = True
# second step: create component pool
SUB_EXP_REPLACEMENT = True  # Set False if already run the code

# Third step: sub component interchange
SEMANTIC_DATA_GENERATION = False

# number of ramdom time for data generating
NUM_AUGMENTED_SAMPLE = 5


WIDTH_PAD = [-100, 100]
# ratio for choosing replacement candidate
RATIO_PAD = [0.8, 1.25]

# local and global distortion

# Config parameters for transformation
G_ANGLE = [-10, 10]
G_SCALE = [-0.1, 0.1]
L_ANGLE = [-10, 10]
L_SCALE = [-0.1, 0.1]
L_TRANSLATE_X = [-5, 5]
L_TRANSLATE_Y = [-0, 0]
S_RATIO = 1

ALL_SYMS = []
PADDING = 10
# ------------------- CONFIGURATION -------------------------------
# Note: This file contain the structure of the HME in the CROHME 2014 training set.
# This data is got after the parsing step with 2D-SCFG algorithm
relation_file = 'relation_data.pkl'

# path to source folder that store Train_2014 dataset folder
DSET_PATH = "./"
# path to saved folder: will save both inkml files and latex list for the inkml
# SAVE_PATH = "D:\\database\\semantic_dataset\\"
SAVE_PATH = "./semantic_dataset/"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

# latex list name
CAPTION_FILE = "dset_caption.txt"
SUB_REP_EXP = "sub_syms_data.pkl"

# load data
sub_data_file = 'sub_data.pkl'
inFp_feature = open(relation_file, 'rb')
data = pkl.load(inFp_feature)
inFp_feature.close()

cap_file = open(SAVE_PATH + CAPTION_FILE, "w+")

type_map = {}
for _type in CONFIG:
    for idx, _term in enumerate(CONFIG[_type]):
        if _type == "SAME":
            type_map[_term] = _type+"%d"%idx
        else:
            type_map[_term] = _type


# ------------------- IMPLEMENTATION -------------------------------
# sub component selector
# config for sub selector
# all_type = ['Expr', 'Expr3', 'Terminal', 'TermLeft', 'LoopList', 'SubExp', 'SupExp', 'UnaryExp', 'ParExp', 'LPExp0', 'FracExp', 'OverExp', 'Term', 'ExpList1', 'Cdot', 'TermRight', 'SetPar', 'RPExp0', 'LoopFactor', 'IntExp1', 'IntPart2', 'IntPart1', 'Num', 'InsExp', 'LmidExp0', 'Int', 'SetExp', 'SetExp1Left', 'LBraceExp0', 'Float', 'Float_Lead', 'RBracketExp0', 'OverExp2', 'FuncExp', 'LI', 'LimExp', 'Lim&RExp', 'ArrowPart1', 'MO', 'Function', 'SumExp', 'SumSub', 'SetExpFull', 'SI', 'SumHorU', 'IntPart4', 'IntPart3', 'FactorialExp', 'SumHorSubU', 'MixFracExp', 'nRoot', 'SetExp1Right', 'Dots1', 'LBracketExp0', 'Expr1', 'NumLoop']
selected_type = ["Expr", "SupExp", "LoopList", "SubExp", "SupExp", "UnaryExp", "ParExp", "FracExp",
                 "LoopFactor", "InsExp", "SetExp", "Float", "FuncExp", "LimExp",
                 "SumExp", "SetExpFull", "FactorialExp", "NumLoop"]

# LimExp, nRoot
def get_strokes(root):
    ret = []
    if "ids" in root:
        return root["ids"]
    if "left" in root:
        ret = ret + get_strokes(root["left"])
    if "right" in root:
        ret = ret + get_strokes(root["right"])

    return ret

# ---- sub --------
def parse_suitable_sub_recursive(root):
    ret = []
    if "type" in root and root["type"] in selected_type:
        ret.append({
            "ids": get_strokes(root),
            "gt": root["gt"],
            "type": "EXPR"
        })
    elif "type" not in root and len(root["gt"].split()) == 1 and type_map[root["gt"]] != "NOT_CHANGE":
        ret.append({
            "ids": get_strokes(root),
            "gt": root["gt"],
            "type": type_map[root["gt"]]
        })
    if "left" in root:
        ret = ret + parse_suitable_sub_recursive(root["left"])
    if "right" in root:
        ret = ret + parse_suitable_sub_recursive(root["right"])
    return ret

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def parse_sub_recursive(root):
    ret = []
    if "ids" in root and root["gt"] not in ALL_SYMS and " " not in root["gt"]:
        ALL_SYMS.append(root["gt"])
    if "type" in root and root["type"] in selected_type:
        added_item = {
            "ids": get_strokes(root),
            "gt": root["gt"],
            "list_rep_ids": [],
            "list_rep_gts": [],
            "list_rep_tmp": [],
            "list_rep_type": [],
        }

        for tmp in ["left", "right"]:
            if tmp in root:
                if "^" not in root[tmp]["gt"] and "_" not in root[tmp]["gt"]  \
                        and (("type" in root[tmp] and root[tmp]["type"] in selected_type) or "type" not in root[tmp]):
                    count_check = added_item["gt"].count(root[tmp]["gt"])
                    if count_check == 1:
                        if " " not in root[tmp]["gt"]:
                            _type_sym = type_map[root[tmp]["gt"]]
                            if _type_sym == "NOT_CHANGE":
                                continue
                            added_item["list_rep_type"].append(_type_sym)
                        else:
                            added_item["list_rep_type"].append("EXPR")

                        added_item["list_rep_gts"].append(root[tmp]["gt"])
                        added_item["list_rep_ids"].append(get_strokes(root[tmp]))
                        tmp_name = "<%s_%d>"%(tmp, len(added_item["list_rep_tmp"]))
                        added_item["list_rep_tmp"].append(tmp_name)
                        added_item["gt"] = added_item["gt"].replace(root[tmp]["gt"], tmp_name)
                else:
                    # add some symbols to be replace
                    suitable_subs = parse_suitable_sub_recursive(root[tmp])

                    # remove duplicate sub and random add
                    random.shuffle(suitable_subs)
                    chosen_exps = []
                    all_ids = []
                    for _sub in suitable_subs:
                        if len(chosen_exps) == 0:
                            if "^" not in _sub["gt"] and "_" not in _sub["gt"]:
                                chosen_exps.append(_sub)
                                all_ids = all_ids + _sub["ids"]
                        else:
                            if len(intersection(all_ids, _sub["ids"])) == 0:
                                if "^" not in _sub["gt"] and "_" not in _sub["gt"]:
                                    chosen_exps.append(_sub)
                                    all_ids = all_ids + _sub["ids"]

                    count_suitable = np.ceil(len(chosen_exps)/4)
                    for _iii, _exp in enumerate(chosen_exps):
                        if count_suitable <= 0:
                            break


                        # check overlap
                        count_check = added_item["gt"].count(_exp["gt"])
                        if count_check == 1 and _exp["type"] != "NOT_CHANGE":
                            added_item["list_rep_type"].append(_exp["type"])
                            added_item["list_rep_gts"].append(_exp["gt"])
                            added_item["list_rep_ids"].append(_exp["ids"])
                            tmp_name = "<%s_%d>" % (_exp["type"], _iii)
                            added_item["list_rep_tmp"].append(tmp_name)
                            added_item["gt"] = added_item["gt"].replace(_exp["gt"], tmp_name)
                            count_suitable -= 1

        ret.append(added_item)

    if "left" in root:
        ret = ret + parse_sub_recursive(root["left"])
    if "right" in root:
        ret = ret + parse_sub_recursive(root["right"])
    return ret

def get_sub_component(data):
    aug_data = []
    for file_name in data:
        exp_data = data[file_name]
        # if file_name == "Train_2014\\105_alfonso.inkml":
        #     print("debug")
        sub_data = parse_sub_recursive(exp_data)
        idx = 0
        for _ii in sub_data:
            _ii["file"] = file_name
            _ii["idx"] = idx
            idx += 1

        aug_data = aug_data + sub_data

    return aug_data

# ---- sub symbols --------
def parse_sub_exp_rep_recursive(root):
    ret = []
    if "^" not in root["gt"] and "_" not in root["gt"] and (("type" in root and root["type"] in selected_type) or ("type" not in root and "ids" in root)) :
        if " " not in root["gt"]:
            _type_sym = type_map[root["gt"]]
            if _type_sym != "NOT_CHANGE":
                ret.append({
                    "ids": get_strokes(root),
                    "gt": root["gt"],
                    "type": type_map[root["gt"]]
                })
        else:
            ret.append({
                "ids": get_strokes(root),
                "gt": root["gt"],
                "type": "EXPR"
            })

    if "left" in root:
        ret = ret + parse_sub_exp_rep_recursive(root["left"])
    if "right" in root:
        ret = ret + parse_sub_exp_rep_recursive(root["right"])
    return ret


def get_sub_exp_replacement_component(data):
    aug_data = []
    for file_name in data:
        exp_data = data[file_name]
        sub_data = parse_sub_exp_rep_recursive(exp_data)
        idx = 0
        for _ii in sub_data:
            _ii["file"] = file_name
            _ii["idx"] = idx
            idx += 1

        aug_data = aug_data + sub_data

    return aug_data

# get list strokes ids to be transformed
def get_stroke_ids(root, head = False):
    ret = []
    left = []
    right = []

    if "left" in root:
        left = get_stroke_ids(root["left"])
    if "right" in root:
        right = get_stroke_ids(root["right"])
    if not head:
        if root["gt"] != "-":
            ret = [get_strokes(root)]
        # else:
        #     print("here")
    else:
        ret = []
    return ret + left + right


# sub component selector
def find_corresponding_part(exps, type, gt, file_name, bbox):
    random.shuffle(exps)
    for item in exps:
        if item["type"] == type and len(gt.split()) == len(item["gt"].split()) \
                and bbox[0]+WIDTH_PAD[0] <= item["width"] <= bbox[0]+WIDTH_PAD[1] and RATIO_PAD[0] <= bbox[1]/item["ratio"] <= RATIO_PAD[1]:
            return item

    return None

# ------------------- RUNNING -------------------------------
if SUB_SPLITER:
    aug_data = get_sub_component(data=data)
    if FILTER:
        aug_data = [item for item in aug_data if FILTER_STR in item["file"]]

    oupFp_feature = open(sub_data_file, 'wb')
    pkl.dump(aug_data, oupFp_feature)
    oupFp_feature.close()
    print("total: %d expr" % (len(aug_data)))

if SUB_EXP_REPLACEMENT:
    sub_syms_exps = get_sub_exp_replacement_component(data=data)
    sub_syms_exps_data = []
    for item in sub_syms_exps:
        if FILTER and FILTER_STR in item["file"]:
            # check ratio
            # load rep_inkml
            rep_inkml_file_name = DSET_PATH + item["file"]
            rep_inkml_obj = Inkml(rep_inkml_file_name)

            # filter selected stroke
            rep_inkml_obj.filter_strokes(ids=item["ids"], new_truth=item["gt"])
            rep_bb = rep_inkml_obj.get_bound_box(selected_ids=item["ids"])
            # item["min_x"] = rep_bb[0]
            # item["min_y"] = rep_bb[1]
            item["width"] = rep_bb[2]
            item["height"] = rep_bb[3]

            if item["height"] > 0 and item["width"] > 0:
                item["ratio"] = rep_bb[2]/ rep_bb[3]
                sub_syms_exps_data.append(item)
            else:
                print(item["file"])
    oupFp_feature = open(SUB_REP_EXP, 'wb')
    pkl.dump(sub_syms_exps_data, oupFp_feature)
    oupFp_feature.close()

count = 0
if SEMANTIC_DATA_GENERATION:
    pkl_file = open(sub_data_file, 'rb')
    aug_data = pkl.load(pkl_file)
    pkl_file.close()

    pkl_file = open(SUB_REP_EXP, 'rb')
    sub_syms_exps = pkl.load(pkl_file)
    pkl_file.close()

    with tqdm(
            total=len(aug_data),
            dynamic_ncols=True,
            leave=True,
    ) as pbar:
        for ii, item in enumerate(aug_data):
            for _idx in range(NUM_AUGMENTED_SAMPLE):
                count_loop = 0
                _ink_selected_ids = list(item["ids"])
                # load inkml file
                inkml_file_name = DSET_PATH + item["file"]
                inkml_obj = Inkml(inkml_file_name)

                # filter selected stroke
                inkml_obj.filter_strokes(ids=_ink_selected_ids, new_truth=item["gt"])

                # load sub inkml file and replace
                new_gt = item["gt"]
                old_ids = []
                new_ids = []
                map_changed = {}
                for idx in range(len(item["list_rep_ids"])):
                    # check which part to be replace
                    rep_ids = item["list_rep_ids"][idx]
                    rep_tmp_name = item["list_rep_tmp"][idx]
                    rep_gt = item["list_rep_gts"][idx]
                    rep_type = item["list_rep_type"][idx]

                    # get bb of the replacement part
                    origin_bb = inkml_obj.get_bound_box(selected_ids=rep_ids)
                    if origin_bb[3] > 0 :
                        origin_ratio = origin_bb[2] / origin_bb[3]
                    else:
                        origin_ratio = 100000

                    # find the corresponding part
                    rep_part = None
                    while rep_part is None:
                        rep_part = find_corresponding_part(exps=sub_syms_exps, type=rep_type, gt=rep_gt, file_name = item["file"], bbox=[origin_bb[2], origin_ratio])
                        count_loop += 1
                        if count_loop > 20:
                            break
                    if count_loop > 20:
                        break

                    # load rep_inkml
                    rep_inkml_file_name = DSET_PATH + rep_part["file"]
                    rep_inkml_obj = Inkml(rep_inkml_file_name)

                    # filter selected stroke
                    rep_inkml_obj.filter_strokes(ids=rep_part["ids"], new_truth=rep_part["gt"])

                    # Transformation
                    # scale transformation
                    # get bouding box for each
                    rep_bb = rep_inkml_obj.get_bound_box(selected_ids=rep_part["ids"])

                    s_x = origin_bb[2] / rep_bb[2] if rep_bb[2] > 0 else 1
                    s_y = origin_bb[3] / rep_bb[3] if rep_bb[3] > 0 else 1

                    rep_inkml_obj.transformation(selected_ids=rep_part["ids"], scale_factor=S_RATIO * np.sqrt(s_x* s_y) if S_RATIO * np.sqrt(s_x* s_y) > 0 else S_RATIO)

                    # translate transformation
                    # get bouding box for each
                    rep_bb = rep_inkml_obj.get_bound_box(selected_ids=rep_part["ids"])

                    t_x = origin_bb[0] - rep_bb[0]
                    t_y = origin_bb[1] - rep_bb[1]

                    rep_inkml_obj.transformation(selected_ids=rep_part["ids"], tx_factor=t_x, ty_factor=t_y)

                    # swap two component
                    max_sid, max_segid = inkml_obj.get_sid_segid()
                    max_sid += 100
                    map_ids = rep_inkml_obj.update_sid_segid(max_sid, max_segid)
                    for _ii in map_ids:
                        # if _ii == "31":
                        #     print("here")
                        new_ids.append(map_ids[_ii])
                    for _ii in rep_ids:
                        old_ids.append(_ii)

                    map_changed[rep_ids[0]] = list(map_ids.values())
                    # swap ink, gt
                    new_gt = new_gt.replace(rep_tmp_name, rep_part["gt"])
                    inkml_obj.swap_ink_object(ori_ids=rep_ids, new_ink_objs=rep_inkml_obj, new_gt=new_gt)

                if count_loop > 20:
                    continue
                # update stroke id
                _new_selected_ids = new_ids
                for _ii in _ink_selected_ids:
                    if _ii not in old_ids:
                        _new_selected_ids.append(_ii)
                _ink_selected_ids = _new_selected_ids

                # get list ids
                list_strokes = get_stroke_ids(data[item["file"]], head=True)

                # check matched ids
                _selected_list = []
                set1 = set(_ink_selected_ids)
                for _ll in list_strokes:
                    _list_id = []
                    for _ii in _ll:
                        if _ii not in old_ids:
                            _list_id.append(_ii)
                        else:
                            if _ii in map_changed:
                                for _jj in map_changed[_ii]:
                                    _list_id.append(_jj)
                    set2 = set(_list_id)
                    if set2.issubset(set1) and len(_list_id):
                        _selected_list.append(_list_id)
                list_strokes = _selected_list

                # local transformation: stroke level and sub_expression level
                for _ll in list_strokes:
                    # random params for local transformation
                    _l_af = random.uniform(L_ANGLE[0] / len(_ll) * 2, L_ANGLE[1] / len(_ll) * 2)
                    _l_sf = random.uniform(L_SCALE[0] / len(_ll) * 2, L_SCALE[1] / len(_ll) * 2) + 1
                    _l_txf = random.uniform(L_TRANSLATE_X[0], L_TRANSLATE_X[1])
                    _l_tyf = random.uniform(L_TRANSLATE_Y[0], L_TRANSLATE_Y[1])

                    # do local transformation
                    inkml_obj.transformation(selected_ids=_ll, angle_factor=_l_af,
                                             scale_factor=_l_sf, tx_factor=_l_txf, ty_factor=_l_tyf)

                # random params for global transformation
                _af = random.uniform(G_ANGLE[0] / len(item["ids"] * 2), G_ANGLE[1] / len(item["ids"] * 2))
                _sf = random.uniform(G_SCALE[0] / len(item["ids"] * 2), G_SCALE[1] / len(item["ids"] * 2)) + 1

                # global transformation
                inkml_obj.transformation(selected_ids=_ink_selected_ids, angle_factor=_af, scale_factor=_sf)

                # re-compute the cords of stroke data
                inkml_obj.re_arrange_stroke_value(selected_ids=_ink_selected_ids, padd=PADDING)

                gt_file_name = "%d_%d.inkml" % (ii, _idx)
                inkml_obj.getInkML(file=SAVE_PATH + gt_file_name)
                cap_file.write("%s\t%s\n" % (gt_file_name, new_gt))

                count += 1
            pbar.update(1)

    print("total: %d files / %d exprs" % (count, len(aug_data)))
    cap_file.close()
