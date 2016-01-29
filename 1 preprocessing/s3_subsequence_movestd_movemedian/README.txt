This step is a Python script which calls through to a compiled binary. The Python script depends on [Bottleneck](https://pypi.python.org/pypi/Bottleneck) (`pip install bottleneck`).
Compile `moving_median_with_mask_function.cpp` to get `moving_median_with_mask_function.exe`.
Run `moving_standard_deviation_mask_occlusion.py`. You may need to change some paths and the keyframe indices inside.

This code uses the keyframe mask sequence and colorshift image sequence resulting from the previous step (2). It output a new image sequence with occlusions removed.
