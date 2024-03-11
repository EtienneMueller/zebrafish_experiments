#import imageio
import numpy as np
#import rawpy
import zarr

from pathlib import Path
from tqdm import tqdm


"""
Script for turning a raw file into a zarr dataset.
Will generate a zebrafish_out.n5 folder containing the subfolders
/volumes/s17/raw/.
reformat_dataset has to be run next.

(Needed around 20 minutes and 182 GB of RAM (=filesize) on HPC for S17)
"""

path = Path(
    #"/nrs/funke/pattonw/data/zebrafish/stitched/top_left_right_bottom_resliced_8555x5155x4419.raw"
    "/data/projects/punim2142/zebrafish_experiments/data/top_left_right_bottom_resliced_8555x5155x4419.raw"  
    #Sample17_top_left_right_bottom_0.0_0.0_0.0_4419x5155x8555_rotated_8555x5155x4419
)

# WILLS LINK IS dataset_container (OUTPUT) IN THE CONSTANTS.YAML
# ‘r’  read only (must exist); ‘r+’ read/write (must exist);
# ‘a’  read/write (create if doesn’t exist);
# ‘w’  create (overwrite if exists); ‘w-’ create (fail if exists).
container = zarr.open(
    #"/nrs/funke/pattonw/predictions/zebrafish/zebrafish.n5", mode="r+"
    "/data/projects/punim2142/zebrafish_experiments/data/zebrafish_out.n5", mode="a"
)
output_data = container.create_dataset(
    "/volumes/s17/raw", dtype=np.uint8, overwrite=True, shape=(8555, 5155, 4419)
)

size_x = 4419
size_y = 5155
size_z = 8555

n_z = 8555
count = size_x * size_y
start_z = 0
end_z = size_z
n_bytes = 1  # Number of bytes in a uint8 (for offset)

fd = open(path, "rb")
fd.seek(start_z)

for i in tqdm(range(start_z, end_z, n_z), desc=path.stem):
    offset = 0
    data = (
        np.fromfile(fd, dtype="b", offset=offset, count=n_z * count)
        .reshape(size_x, size_y, -1)
        .transpose((2, 1, 0))
    )
    output_data[i : i + data.shape[0]] = data
fd.close()
