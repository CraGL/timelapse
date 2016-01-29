# Decomposing Time-Lapse Paintings into Layers

This code implements the pipeline described in the paper "Decomposing Time-Lapse Paintings into Layers" by Jianchao Tan, Marek Dvorožňák, Daniel Sýkora, Yotam Gingold from SIGGRAPH 2015. The pipeline is divided into two stages.

### 1 Preprocessing:
- Input: raw time-lapse video
- Output: albedo video

The substeps are:

1. Color shift the whole sequence
2. Extract keyframes and color shift each sub-sequence
3. For each sub-sequence, perform moving std. deviation and moving median
4. Whole sequence L0 smoothing
5. Perform albedo conversion

### 2 Layer extraction
- Input: albedo video
- Output: KM layers and PD layers

The programs are:

- PD layer extraction and KM layer extraction
- PD using the spatial coherency solution: The 3-by-3 layer extraction described in the paper

## Dependencies

- OpenCV 2.4
- Eigen 3
- JsonCpp 0.5
- zlib
- [Bottleneck](https://pypi.python.org/pypi/Bottleneck): `pip install bottleneck`
- PIL or Pillow (Python Image Library): `pip install Pillow`
- NumPy
- LAPACK
