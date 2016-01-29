#!/usr/local/bin/python2.7

from numpy import *
## Only Pillow (PIL) works for Jianchao on Windows
from PIL import Image

import os, sys

def usage():
    print >> sys.stderr, 'Usage:', sys.argv[0], '[--glob] [--max-GB 2] spatial_pixels_per_timewise_image path/to/image1 [path/to/image2 ... path/to/imageN] path/to/output_basename'
    sys.exit(-1)

argv = list( sys.argv[1:] )

glob_filenames = False
try:
    i = argv.index( '--glob' )
    glob_filenames = True
    del argv[ i ]
except ValueError: pass

## The maximum memory buffer in GB
max_GB = 2
try:
    i = argv.index( '--max-GB' )
    max_GB = float( argv[ i + 1 ] )
    del argv[ i : i + 2 ]
except IndexError: usage()
except ValueError: pass

if len( argv ) < 3:
    usage()

spatial_pixels_per_output = int( argv.pop(0) )
filenames = argv[:-1]
outname_basename = argv[-1]

if glob_filenames:
    from glob import glob
    from itertools import chain
    filenames = list( chain( *[ glob( fname ) for fname in filenames ] ) )

## Check this later, so that the image data will be printed first.
#assert max_pixels_per_image >= len( filenames )

## TODO: I can't assert this yet.
# if os.path.exists( outname ):
#    print >> sys.stderr, 'ERROR: Output file exists, will not clobber:', outname
#    usage()

image_shape = asarray( Image.open( filenames[0] ).convert( 'RGB' ), dtype = uint8 ).shape
# spatial_pixels_per_output = min( max_pixels_per_image // len( filenames ), prod( image_shape[:2] ) )
print 'Image dimensions:', image_shape[0], 'by', image_shape[1], 'and', len( filenames ), 'frames.'
print 'Each time-wise image will have', spatial_pixels_per_output, 'of the pixels (row-major)',
num_images = (prod( image_shape[:2] )+spatial_pixels_per_output-1)//spatial_pixels_per_output
print 'for a total of', num_images, 'images.'

if spatial_pixels_per_output < 1:
    print >> sys.stderr, 'ERROR: spatial_pixels_per_timewise_image must be a positive integer.'
    usage()

## 2 GB
max_ram_usage_bytes = int( max_GB*1024*1024*1024 )
max_spatial_pixels_in_memory = max_ram_usage_bytes // ( len(filenames) * image_shape[2] )
spatial_pixels_in_memory = spatial_pixels_per_output * ( max_spatial_pixels_in_memory // spatial_pixels_per_output )
if spatial_pixels_in_memory < spatial_pixels_per_output:
    print >> sys.stderr, 'ERROR: Maximum RAM usage would be exceeded to save even one image',
    print >> sys.stderr, '(which would take', ( spatial_pixels_per_output * len(filenames) * image_shape[2] ), 'bytes).'
    usage()

stack = empty( ( spatial_pixels_in_memory, len(filenames), image_shape[2] ), dtype = uint8 )
count = 0
for mem_off in xrange( 0, prod( image_shape[:2] ), spatial_pixels_in_memory ):
    mem_end = min( mem_off + spatial_pixels_in_memory, prod( image_shape[:2] ) )
    
    for frame_count, fname in enumerate( filenames ):
        print 'Loading "%s"' % ( fname, )
        img = Image.open( fname ).convert( 'RGB' )
        arr = asarray( img, dtype = uint8 )
        arr = arr.reshape( -1, arr.shape[2] )[ mem_off : mem_end ]
        stack[ :mem_end-mem_off, frame_count, : ] = arr
    
    for save_off in xrange( 0, mem_end-mem_off, spatial_pixels_per_output ):
        save_end = min( save_off + spatial_pixels_per_output, mem_end-mem_off )
        
        outname = outname_basename + '-%dx%d-%04d.png' % ( image_shape[0], image_shape[1], count )
        Image.fromarray( stack[ save_off:save_end ] ).save( outname )
        print 'Saved', outname
        
        count += 1
