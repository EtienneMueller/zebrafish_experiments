import daisy
import logging
import numpy as np
import os
import time

from funlib.segment.graphs.impl import connected_components
from funlib.persistence import graphs
from funlib.persistence import open_ds


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("funlib..persistence.graphs.shared_graph_provider").setLevel(
    logging.DEBUG
)


def find_segments(
    db_host,
    db_name,
    fragments_file,
    fragments_dataset,
    edges_collection,
    thresholds_minmax,
    thresholds_step,
    sample_name,
):
    """
    Args:

        db_host (``string``):
            Where to find the MongoDB server.

        db_name (``string``):
            The name of the MongoDB database to use.

        fragments_file (``string``):
            Path to the file containing the fragments.

        edges_collection (``string``):
            The name of the MongoDB database collection to use.

        thresholds_minmax (``list`` of ``int``):
            The lower and upper bound to use (i.e [0,1]) when generating
            thresholds.

        thresholds_step (``float``):
            The step size to use when generating thresholds between min/max.

        roi_offset (array-like of ``int``):
            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``):
            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.
    """

    scale = False

    print("Reading graph from DB ", db_name, edges_collection)
    start = time.time()

    graph_provider = graphs.MongoDbGraphProvider(
        db_name,
        db_host,
        nodes_collection=f"{sample_name}_nodes",
        meta_collection=f"{sample_name}_meta",
        edges_collection=edges_collection,
        position_attribute=["center_z", "center_y", "center_x"],
    )

    fragments = open_ds(fragments_file, fragments_dataset)

    roi = fragments.roi
    roi = None
    # roi = daisy.Roi(roi.offset, (20000, 10000, 10000))

    graph = graph_provider.get_graph(roi)

    print("Read graph in %.3fs" % (time.time() - start))
    print("Graph contains %d nodes, %d edges" % (len(graph.nodes), len(graph.edges)))

    if graph.number_of_nodes == 0:
        print("No nodes found in roi %s" % roi)
        return

    nodes = np.array(graph.nodes)
    edges = np.stack(list(graph.edges), axis=0)
    scores = np.array([graph.edges[tuple(e)]["merge_score"] for e in edges]).astype(
        np.float32
    )

    print("Nodes dtype: ", nodes.dtype)
    print("edges dtype: ", edges.dtype)
    print("scores dtype: ", scores.dtype)

    print("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))

    out_dir = os.path.join(fragments_file, "luts_full", "fragment_segment")

    os.makedirs(out_dir, exist_ok=True)

    thresholds = list(
        np.arange(thresholds_minmax[0], thresholds_minmax[1], thresholds_step)
    )

    start = time.time()

    for threshold in thresholds:
        get_connected_components(
            nodes,
            edges,
            scores,
            threshold,
            edges_collection,
            out_dir,
        )

        print("Created and stored lookup tables in %.3fs" % (time.time() - start))


def get_connected_components(
    nodes, edges, scores, threshold, edges_collection, out_dir, **kwargs
):
    # drop out mask will be added to scores. anything with a 1 added will always be above threshold
    # dropout_mask = np.random.randn(len(scores)) / 500
    # dropout_mask = dropout_mask.astype(np.float32)
    # scores += dropout_mask

    print("Getting CCs for threshold %.3f..." % threshold)
    start = time.time()
    components = connected_components(nodes, edges, scores, threshold)
    print("%.3fs" % (time.time() - start))

    print("Creating fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()
    lut = np.array([nodes, components])

    print("%.3fs" % (time.time() - start))

    print("Storing fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()

    lookup = "seg_%s_%d" % (edges_collection, int(threshold * 10000))
    lookup = lookup.replace("/", "-")

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(
        out_file, fragment_segment_lut=lut, edges=edges, merged_edges=scores < threshold
    )

    print("%.3fs" % (time.time() - start))


if __name__ == "__main__":
    start = time.time()
    find_segments(
        db_host="mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017",
        db_name="dacapo_zebrafish",
        fragments_file="/nrs/funke/pattonw/predictions/zebrafish/zebrafish.n5",
        fragments_dataset=f"predictions/2023-05-09/s17/cells_finetuned_3d_lsdaffs_zebrafish_cells_upsample-unet_default_v3__1__60000_fragments",
        edges_collection=f"s17-stitched_edges_hist_quant_75",
        thresholds_minmax=[0.0, 1.0],
        thresholds_step=0.05,
        sample_name=f"s17-stitched",
    )
    print("Took %.3f seconds to find segments and store LUTs" % (time.time() - start))
