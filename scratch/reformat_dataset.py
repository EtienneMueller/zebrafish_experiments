"""Script for converting the various training data into a consistent format for
training. Requires a significant amount of RAM to Run"""
import click
import csv
import itertools
import logging
import yaml
import zarr

import numpy as np

from pathlib import Path


# Load yamls
yaml_root_dir = Path("configs/yamls/zebrafish")
assert yaml_root_dir.exists(), f"{yaml_root_dir} does not exist!"

# constants
constants_yaml = yaml_root_dir / "constants.yaml"
constants = yaml.safe_load(constants_yaml.open("r").read())

# targets. What classes we need to preprocess the data into
target_yaml = yaml_root_dir / "targets.yaml"
targets = yaml.safe_load(target_yaml.open("r").read())

# dataset yamls. What crops do we have per dataset, and where is the raw data
dataset_yaml = yaml_root_dir / "datasets.yaml"
datasets = yaml.safe_load(dataset_yaml.open("r").read())

# id annotations
id_annotations_yaml = yaml_root_dir / "id_annotations.yaml"
id_annotations = yaml.safe_load(id_annotations_yaml.open("r").read())

in_container = zarr.open(constants["input_container"])
out_container = zarr.open(constants["dataset_container"])

logger = logging.getLogger(__name__)


def setup_logging(level):
    logging.basicConfig(
        format='%(asctime)s-[%(lineno)d]-%(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        level=level,
    )
    logger.debug("Logging is set to debug mode.")


def copy_data(crop, index, organelle):
    logging.info(f"COPY_DATA: {crop = }; {index = }; {organelle = }")
    
    raw = in_container[constants["raw_dataset"].format(sample=f"{crop}-{index}")][:]
    out_container[
        constants["raw_dataset"].format(sample=f"{crop}-{index}")
    ] = raw.astype(np.uint8)

    try:
        logging.debug(f"try: check for gt_dataset")
        in_data = in_container[
            constants["gt_dataset"].format(
                sample=f"{crop}-{index}", organelle=organelle
            )
        ][:]
    except KeyError as e:
        logging.debug(f"except: KeyError")
        try:
            logging.debug(f"try: check for gt_dataset, organelle 'all'")
            in_data = in_container[
                constants["gt_dataset"].format(
                    sample=f"{crop}-{index}", organelle="all"
                )
            ][:]
        except KeyError as e:
            logging.debug(f"except: KeyError")
            # assume organelle not present in this crop
            in_data = np.zeros(raw.shape)

    id_mapping = (
        constants["data_problems"]
        .get(f"{crop}-{index}", {})
        .get(organelle, {})
        .get("id_mapping", [])
    )
    logging.debug(f"{id_mapping = }")
    for bad_id, new_id in (id_mapping):
        in_data[in_data == bad_id] = new_id
    
    mask_id = (
        constants["data_problems"]
        .get(f"{crop}-{index}", {})
        .get(organelle, {})
        .get("mask_id", None)
    )
    logging.debug(f"{mask_id = }")

    if mask_id is not None:
        in_mask = in_data != (
            constants["data_problems"]
            .get(f"{crop}-{index}", {})
            .get(organelle, {})
            .get("mask_id", -1)
        )
    else:
        logging.debug(f"else: mask_id IS None")
        logging.debug(f"in_data shape is {np.shape(in_data)}")
        in_mask = np.ones_like(in_data)
    in_data[in_mask == 0] = 0

    out_container[
        constants["gt_dataset"].format(sample=f"{crop}-{index}", organelle=organelle)
    ] = in_data.astype(np.uint64)
    out_container[
        constants["mask_dataset"].format(sample=f"{crop}-{index}", organelle=organelle)
    ] = in_mask.astype(np.uint64)


def update_masks(crop, index, targets):
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
    print(organelle, sample, annotation_type)
    if annotation_type == "good":
        pass
    elif annotation_type == "negative":
        return
    elif annotation_type == "segments":
        return
    else:
        return

    with open(
        constants["id_annotations_path"].format(
            organelle=organelle, sample=sample, annotation_type=annotation_type
        ),
        newline="",
    ) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if annotation_type == "good":
        row = rows[0]
        good_ids = [int(x) for x in row]
    elif annotation_type == "negative":
        row = rows[0]
        bad_ids = [int(x) for x in row]
    elif annotation_type == "segments":
        segments = [[int(y) for y in x if len(y) > 0] for x in zip(*rows)]

    fragments = in_container[
        constants["gt_dataset"].format(sample=sample, organelle=organelle)
    ][:]
    try:
        in_gt = out_container[
            constants["gt_dataset"].format(sample=sample, organelle=organelle)
        ][:]
    except KeyError:
        in_gt = np.zeros_like(fragments)
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

    print(f"Mask sum: {in_mask.sum()}")
    out_container[
        constants["gt_dataset"].format(sample=sample, organelle=organelle)
    ] = in_gt.astype(np.uint64)
    out_container[
        constants["mask_dataset"].format(sample=sample, organelle=organelle)
    ] = in_mask.astype(np.uint64)


def merge_masks(organelle_a: str, organelle_b: str, sample: str):
    try:
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
        in_gt_a = None
        in_mask_a = None
    try:
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
        in_gt_b = None
        in_mask_b = None

    a_masks = ([in_mask_a] if in_mask_a is not None else []) + (
        [in_gt_b] if in_gt_b is not None else []
    )
    b_masks = ([in_mask_b] if in_mask_b is not None else []) + (
        [in_gt_a] if in_gt_a is not None else []
    )
    in_mask_a = np.stack(a_masks).max(axis=0)
    in_mask_b = np.stack(b_masks).max(axis=0)

    out_container[
        constants["mask_dataset"].format(sample=sample, organelle=organelle_a)
    ] = in_mask_a.astype(np.uint64)
    out_container[
        constants["mask_dataset"].format(sample=sample, organelle=organelle_b)
    ] = in_mask_b.astype(np.uint64)


def generate_points(sample: str, organelle: str):
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


@click.command()
@click.option('--copy-data', 'copy_data_flag', is_flag=True, help='Copy data.')
@click.option('--relabel', 'relabel_flag', is_flag=True, help='Relabel dataset.')
@click.option('--merge-masks', 'merge_masks_flag', is_flag=True, help='Merge masks.')
@click.option('--update-masks', 'update_masks_flag', is_flag=True, help='Ipdate masks.')
@click.option('--generate-points', 'generate_points_flag', is_flag=True, help='generate points.')
@click.option('--log-level', default='INFO', help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).')
def main(copy_data_flag, relabel_flag, merge_masks_flag, update_masks_flag, generate_points_flag, log_level):
    level = getattr(logging, log_level.upper(), logging.INFO)
    setup_logging(level)

    # condition = lambda sample, organelle, annotation_type: (
    #     organelle == "cells" and sample == "23_mid1"
    # )
    logging.info(f"Running copy_data: {copy_data_flag}")
    logging.info(f"Running relabel: {relabel_flag}")
    logging.info(f"Running merge_masks: {merge_masks_flag}")
    logging.info(f"Running update_masks: {update_masks_flag}")
    logging.info(f"Running generate_points: {generate_points_flag}")

    if copy_data_flag:
        for crop, index in [[8, 1], [8, 2], [17, 2]]:  #, [16, "bot"], [23, "mid1"]]:
            copy_data(crop, index, "cells")
        for organelle in ["vessel", "axons"]:
            for crop, index in [[8, 1], [8, 2]]:  #, [23, "bot"]]:
                copy_data(crop, index, organelle)


    if relabel_flag:
        for organelle, samples in id_annotations.items():
            for sample, annotation_types in samples.items():
                for annotation_type in annotation_types:
                    if not condition(sample, organelle, annotation_type):
                        continue
                    relabel(organelle, sample, annotation_type)

    if merge_masks_flag:
        for organelle_a, organelle_b in itertools.combinations(id_annotations.keys(), 2):
            organelle_a_samples = set(id_annotations[organelle_a].keys())
            organelle_b_samples = set(id_annotations[organelle_b].keys())
            for sample in organelle_a_samples.union(organelle_b_samples):
                if not (
                    condition(sample, organelle_a, None)
                    or condition(sample, organelle_b, None)
                ):
                    continue
                merge_masks(organelle_a, organelle_b, sample)

    if update_masks_flag:
        for crop, index in itertools.chain(datasets["train"], datasets["validate"]):
            if not condition(None, None, None):
                continue
            update_masks(crop, index, targets)

    if generate_points_flag:
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
            if not condition(sample, organelle, None):
                continue
            generate_points(sample, organelle)


if __name__ == "__main__":
    main()
