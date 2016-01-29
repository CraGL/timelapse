This step depends on NumPy and LAPACK and PIL (or Pillow).

First, run `timerotate.py` to convert an image sequence (the result of step 3) to a set of timewise images (each row is a pixel across time). See `rose-timewise.HOW`.

Then, run `l0smooth1Dsparse.py`, to get l0-smoothed timewise images. See `l0smooth.HOW`.

Finally, run `timeunrotate.py` to convert the l0-smoothed timewise images back to an image sequence. See `rose-unrotate.HOW`.
