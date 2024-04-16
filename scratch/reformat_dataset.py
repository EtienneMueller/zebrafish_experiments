import csv
import itertools
import logging
import numpy as np
import sys
import yaml
import zarr
from pathlib import Path


# Script for converting the various training data into a consistent
# format for training. Requires a significant amount of RAM to Run

"""
Description
sample = {crop}-{index} e.g. 16_bot, 23_bot, 23_mid1
organelle = axon, cell, vessel

This script consists of five major part (in order to run):
copy_data, relabel, merge_masks/update_masks, generate_points

copy_data
    Will: "copy_data copies the raw data from zebrafish.n5 and does any label
    mapping (getting rid of bad labels, separating out various mask labels into
    separate arrays)"

relabel
    Will: "relabel takes in the predictions plus the lists of "good"/"bad"/
    "negative" IDs and creates a new gt/mask arrays"

merge_masks/update_masks
    Will: "merge_masks/update_masks do the same thing, I think merge_masks was
    taking too long so I re-implemented it a bit faster. Is just augmenting the
    masks for each organelle with the ground truth from other organelles.
    I.e.cells won't be vessels so you can mask in all the cells in the vessel
    training data."

generate_points
    Will: "generate_points generates a bunch of points for sample randomly for
    training. This is only necessary for the very large very sparse training
    datasets."
"""

# Load yamls
yaml_root_dir = Path("/data/projects/punim2142/zebrafish_experiments/configs/yamls/zebrafish")
assert yaml_root_dir.exists(), f"{yaml_root_dir} does not exist!"

# constants
constants_yaml = yaml_root_dir / "constants.yaml"
constants = yaml.safe_load(constants_yaml.open("r").read())

# targets.
# What classes we need to preprocess the data into
# (vessels, axons, cells)
target_yaml = yaml_root_dir / "targets.yaml"
targets = yaml.safe_load(target_yaml.open("r").read())

# dataset yamls.
# What crops do we have per dataset, and where is the raw data
# (train: 17-2, 8-1, 8-2, 16_bot, 23_bot, 23_mid1, validate: 8-2)
dataset_yaml = yaml_root_dir / "datasets.yaml"
datasets = yaml.safe_load(dataset_yaml.open("r").read())

# id annotations
# (axons: 23_bot: bad, negative, segments)
# (cells: 16_bot: bad, negative, good; 23_mid1: bad, negative, good)
# (vessel: 23_bot: negative, segments)
id_annotations_yaml = yaml_root_dir / "id_annotations.yaml"
id_annotations = yaml.safe_load(id_annotations_yaml.open("r").read())

in_container = zarr.open(constants["input_container"])
out_container = zarr.open(constants["dataset_container"])


def copy_data(crop, index, organelle):
    """
    (Needed 140 GB and 12 min for 23-mid1)
    (Needed 140 GB and 51 min for 16-bot, 23-bot, 23-mid1, 23-top)

    Arguments:
        crop {}: e.g. 17, 8, 16, 23

        index {str}: e.g. 1, 2, mid1, bot

        organelle {str}: e.g. cells, axon, vessel
    """
    # here: zebrafish.n5/volumes/s17/raw/
    raw = in_container[constants["raw_dataset"].format(sample=f"{crop}_{index}")][:]
    # here: zebrafish_out.n5/volumes/23-mid1/raw/
    out_container[
        constants["raw_dataset"].format(sample=f"{crop}-{index}")
    ] = raw.astype(np.uint8)

    try:
        logging.info(f"trying {constants['gt_dataset']} {organelle}")
        # here: zebrafish.n5/volumes/23-mid1/vessel
        # DOES NOT EXIST
        in_data = in_container[
            constants["gt_dataset"].format(
                sample=f"{crop}_{index}", organelle=organelle
            )
        ][:]
    except KeyError as e:
        logging.info("KeyError")
        try:
            logging.info(f"Try gt_dataset2 {constants['gt_dataset']} all")
            # here: zebrafish.n5/volumes/23-1/vessel
            # DOES NOT EXIST
            in_data = in_container[
                constants["gt_dataset"].format(
                    sample=f"{crop}_{index}", organelle="all"
                )
            ][:]
        except KeyError as e:
            # assume organelle not present in this crop
            logging.info("KeyError")
            logging.info("Assuming organelle not present in this crop")
            logging.info("Setting in_data to all zeros")
            in_data = np.zeros(raw.shape)

    # Crop 23-1 does not exist, nothing will happen?????
    logging.info("in_data = new_id")
    for bad_id, new_id in (
        constants["data_problems"]
        .get(crop, {})
        .get(index, {})
        .get(organelle, {})
        .get("id_mapping", [])
    ):
        in_data[in_data == bad_id] = new_id

    # Still no 23-1
    logging.info("mask_id")
    mask_id = (
        constants["data_problems"]
        .get(crop, {})
        .get(index, {})
        .get(organelle, {})
        .get("mask_id", None)
    )

    # Still no 23-1
    # therefore ELSE
    logging.info("if mask_id is not None")
    if mask_id is not None:
        logging.info("true")
        in_mask = in_data != (
            constants["data_problems"]
            .get(crop, {})
            .get(index, {})
            .get(organelle, {})
            .get("mask_id", -1)
        )
    else:
        logging.info("else")
        in_mask = np.ones_like(in_data)
    
    logging.info("in_data[in_mask == 0] = 0")
    in_data[in_mask == 0] = 0

    # here: zebrafish_out.n5/volumes/23-1/vessel/
    logging.info("out_container gt_dataset")
    out_container[
        constants["gt_dataset"].format(sample=f"{crop}-{index}", organelle=organelle)
    ] = in_data.astype(np.uint64)
    
    # here: zebrafish_out.n5/volumes/23-1/vessel_mask/
    logging.info("out_container mask_dataset")
    out_container[
        constants["mask_dataset"].format(sample=f"{crop}-{index}", organelle=organelle)
    ] = in_mask.astype(np.uint64)


def update_masks(crop, index, targets):
    """
    Arguments:
        organelle {str}: e.g. cells, axon, vessel
        
        sample {str}: e.g. 23_bot

        annotation_type {str}: e.g. good, negative, segments
    """
    # any annotated organelle can be masked in for the other organelle training volumes
    annotated_organelles = (
        sum(
            [
                out_container[
                    constants["gt_dataset"].format(
                        sample=f"{crop}-{index}", organelle=organelle
                    )
                ][:]
                for organelle in targets
            ]
        )
        > 0
    )
    for organelle in targets:
        mask_dataset = out_container[
            constants["mask_dataset"].format(
                sample=f"{crop}-{index}", organelle=organelle
            )
        ]
        mask_dataset[:] = (mask_dataset + annotated_organelles) > 0


def relabel(organelle: str, sample: str, annotation_type: str):
    """
    Super fast: 1s and 1MB

    Arguments:
        organelle {str}: e.g. cells, axon, vessel
        
        sample {str}: e.g. 23_bot

        annotation_type {str}: e.g. good, negative, segments
    """

    # only continue if annotation_type is good (WHY?)
    if annotation_type == "good":
        logging.info("annotation_type = good")
        pass
    elif annotation_type == "negative":
        logging.info("annotation_type = negative: SKIP")
        return
    elif annotation_type == "segments":
        logging.info("annotation_type = segments: SKIP")
        return
    else:
        logging.info("annotation_type = else: SKIP")
        return

    # id_annotations.yaml
    with open(
        constants["id_annotations_path"].format(
            organelle=organelle, sample=sample, annotation_type=annotation_type
        ),
        newline="",
    ) as f:
        reader = csv.reader(f)
        rows = list(reader)

    logging.info(f"Opening id_path + {organelle} {sample} {annotation_type}")
    logging.info(f"{rows[0]}")

    if annotation_type == "good":
        row = rows[0]
        good_ids = [int(x) for x in row]
    elif annotation_type == "negative":  # is skipped anyways
        row = rows[0]
        bad_ids = [int(x) for x in row]
    elif annotation_type == "segments":  # is skipped anyways
        segments = [[int(y) for y in x if len(y) > 0] for x in zip(*rows)]

    # gt_dataset ("volumes/{sample}/{organelle}") from INPUT_CONTAINER
    # PROBLEM HERE
    # I don't have the zebrafish.n5 file mentioned
    # Tried to recreate it but it is not working so far
    # Should look something like this:
    # zebrafish.n5
    # |-23_bot
    # |--cells
    fragments = in_container[
        constants["gt_dataset"].format(sample=sample, organelle=organelle)
    ][:]

    # in_gt is set to whatever is saved in dataset_container
    # HOW DOES IT GET THERE?
    # if it does not exist it gets filled with zeros
    try:
        in_gt = out_container[
            constants["gt_dataset"].format(sample=sample, organelle=organelle)
        ][:]
    except KeyError:
        in_gt = np.zeros_like(fragments)
    
    # in_mask is set to whatever is saved in dataset_container
    # HOW DOES IT GET THERE?
    # if it does not exist it gets filled with zeros
    try:
        in_mask = out_container[
            constants["mask_dataset"].format(sample=sample, organelle=organelle)
        ][:]
    except KeyError:
        in_mask = np.zeros_like(fragments)

    if annotation_type == "good":
        row = rows[0]
        good_ids = [int(x) for x in row]
        mask = np.isin(fragments, good_ids)
        in_gt[mask] = fragments[mask]
        in_mask = np.stack([in_mask, mask]).max(axis=0)
    elif annotation_type == "negative":
        bad_ids = [int(x) for x in row]
        mask = np.isin(fragments, bad_ids)
        in_mask = np.stack([in_mask, mask]).max(axis=0)
    elif annotation_type == "segments":
        for segment in segments:
            segment_id = min(segment)
            mask = np.isin(fragments, segment)
            in_gt[mask] = segment_id
            in_mask = np.stack([in_mask, mask]).max(axis=0)
    else:
        return

    raw = in_container[constants["raw_dataset"].format(sample=sample)][:]
    out_container[constants["raw_dataset"].format(sample=sample)] = raw.astype(np.uint8)

    logging.info(f"Mask sum: {in_mask.sum()}")
    out_container[
        constants["gt_dataset"].format(sample=sample, organelle=organelle)
    ] = in_gt.astype(np.uint64)
    out_container[
        constants["mask_dataset"].format(sample=sample, organelle=organelle)
    ] = in_mask.astype(np.uint64)


def merge_masks(organelle_a: str, organelle_b: str, sample: str):
    """
    Arguments:
        organelle {str}: e.g. cells, axon, vessel
        
        sample {str}: e.g. 23_bot

        annotation_type {str}: e.g. good, negative, segments
    """
    try:
        logging.info("try")
        in_gt_a = (
            out_container[
                constants["gt_dataset"].format(sample=sample, organelle=organelle_a)
            ][:]
            > 0
        )
        in_mask_a = out_container[
            constants["mask_dataset"].format(sample=sample, organelle=organelle_a)
        ][:]
    except KeyError as e:
        logging.info("except")
        in_gt_a = None
        in_mask_a = None
    try:
        logging.info("try2")
        in_gt_b = (
            out_container[
                constants["gt_dataset"].format(sample=sample, organelle=organelle_b)
            ][:]
            > 0
        )
        in_mask_b = out_container[
            constants["mask_dataset"].format(sample=sample, organelle=organelle_b)
        ][:]
    except KeyError as e:
        logging.info("except2")
        in_gt_b = None
        in_mask_b = None

    a_masks = ([in_mask_a] if in_mask_a is not None else []) + (
        [in_gt_b] if in_gt_b is not None else []
    )
    b_masks = ([in_mask_b] if in_mask_b is not None else []) + (
        [in_gt_a] if in_gt_a is not None else []
    )
    if in_mask_a is not None:
        in_mask_a = np.stack(a_masks).max(axis=0)
        out_container[constants["mask_dataset"].format(
            sample=sample,
            organelle=organelle_a)] = in_mask_a.astype(np.uint64)
    if in_mask_b is not None:
        in_mask_b = np.stack(b_masks).max(axis=0)
        out_container[constants["mask_dataset"].format(
            sample=sample,
            organelle=organelle_b)] = in_mask_b.astype(np.uint64)


def generate_points(sample: str, organelle: str):
    """
    Arguments:
        organelle {str}: e.g. cells, axon, vessel
        
        sample {str}: e.g. 23_bot

        annotation_type {str}: e.g. good, negative, segments
    """
    
    in_mask = out_container[
        constants["mask_dataset"].format(sample=sample, organelle=organelle)
    ][:]

    (z, y, x) = np.where(in_mask)

    np.savez(
        constants["dataset_container"]
        + "/"
        + constants["points_dataset"].format(sample=sample, organelle=organelle),
        [z, y, x],
    )


COPY_DATA = True
RELABEL = False
MERGE_MASKS = False
UPDATE_MASKS = False
GENERATE_POINTS = False

logging.basicConfig(
    #format='%(asctime)s %(levelname)-2s [%(lineno)d] %(message)s',
    format='%(asctime)s %(levelname)s [%(lineno)d] %(message)s',
    datefmt='%Y/%m/%d-%H:%M:%S',
    level=logging.INFO
)

condition = lambda sample, organelle, annotation_type: (
    organelle == "cells" and sample == "23-top"
)

# some error in itertools.chain
if COPY_DATA:
    logging.info("-COPY_DATA-")
    for organelle in targets:  # targets = [vessel, axons, cells]
        logging.info(f" {itertools.chain(datasets['train'], datasets['validate'])}")
        #for crop, index in itertools.chain(datasets["train"], datasets["validate"]):
        for crop, index in [[8, 2]]:  #[[17, 2], [8, 1], [8, 2], [8, 2]]:  #[[16, "bot"], [23, "bot"], [23, "mid1"], [23, "top"]]:
            #if not condition(None, organelle, None):
            #    logging.info(f"No copying data: {organelle}, {crop}, {index}"")
            #    continue
            logging.info(f"COPYING DATA: {organelle}, {crop}, {index}")
            copy_data(crop, index, organelle)

# see id_annotations.yaml
if RELABEL:
    logging.info("-RELABEL-")
    for organelle, samples in id_annotations.items():
        for sample, annotation_types in samples.items():
            for annotation_type in annotation_types:
                if not condition(sample, organelle, annotation_type):
                    logging.info(f"Skipping: {organelle}, {sample}, {annotation_type}")
                    continue
                logging.info(f"Relabeling: {organelle}, {sample}, {annotation_type}")
                relabel(organelle, sample, annotation_type)

if MERGE_MASKS:
    logging.info("-MERGE_MASKS-")
    for organelle_a, organelle_b in itertools.combinations(id_annotations.keys(), 2):
        organelle_a_samples = set(id_annotations[organelle_a].keys())
        organelle_b_samples = set(id_annotations[organelle_b].keys())
        for sample in organelle_a_samples.union(organelle_b_samples):
            if not (
                condition(sample, organelle_a, None)
                or condition(sample, organelle_b, None)
            ):
                logging.info(f"No merging masks: {organelle_a}, {organelle_a_samples}, {sample}")
                continue
            logging.info(f"Merge masks: {organelle_a}, {organelle_a_samples}, {sample}")
            merge_masks(organelle_a, organelle_b, sample)

# error
if UPDATE_MASKS:
    logging.info("-UPDATE_MASKS-")
    #for crop, index in itertools.chain(datasets["train"], datasets["validate"]):
    for crop, index in [[23, 1]]:  #[[17, 2], [8, 1], [8, 2], [8,2]]:
        #if not condition(None, None, None):
        #    logging.info(f"No updating masks: {crop}, {index}")
        #    continue
        logging.info(f"Updating masks: {crop}, {index}")
        update_masks(crop, index, targets)

# error
if GENERATE_POINTS:
    logging.info("-GENERATE_MASKS-")
    for sample, organelle in itertools.product(
        [
            "16_bot",
            "23_bot",
            "23_mid1",
        ],
        [
            "axons",
            "cells",
            "vessel",
        ],
    ):
        #logging.info(f"{sample}, {organelle}"")
        if not condition(sample, organelle, None):
            logging.info(f"No generating points: {sample}, {organelle}")
            continue
        logging.info(f"Generating points: {sample}, {organelle}")
        generate_points(sample, organelle)
