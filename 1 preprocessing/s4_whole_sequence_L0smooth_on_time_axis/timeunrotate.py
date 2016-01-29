#!/usr/local/bin/python2.7

from numpy import *
## Only Pillow (PIL) works for Jianchao on Windows
from PIL import Image

import os, sys

def usage():
    print >> sys.stderr, 'Usage:', sys.argv[0], '[--glob] [--max-GB 2] path/to/image1 [path/to/image2 ... path/to/imageN] path/to/output_basename'
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

if len( argv ) < 2:
    usage()

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

output_shape = set([ tuple(list(map( int, filename.split('-')[-2].lower().split('x') ))) for filename in filenames ])
if len( output_shape ) != 1:
    print >> sys.stderr, "Multiple output image shapes (read from the second-to-last _NxM_ part of the filename) detected."
    usage()
output_shape = output_shape.pop()

image_shape = asarray( Image.open( filenames[0] ).convert( 'RGB' ), dtype = uint8 ).shape
output_shape = ( output_shape[0], output_shape[1], image_shape[2] )
# spatial_pixels_per_output = min( max_pixels_per_image // len( filenames ), prod( image_shape[:2] ) )
print 'Time-wise image dimensions:', image_shape[0], 'by', image_shape[1], 'and', len( filenames ), 'frames.'
print 'Each output image will be', output_shape[0], 'by', output_shape[1], 'by', output_shape[2]
#max_output_images = image_shape[0] * len( filenames ) // ( output_shape[0]*output_shape[1] )
num_output_images = image_shape[1]

## 2 GB
max_ram_usage_bytes = int( max_GB*1024*1024*1024 )
## Testing:
# max_ram_usage_bytes = 2*prod( output_shape ) + 10
# num_output_images = 5
max_images_in_memory = max_ram_usage_bytes // prod( output_shape )
images_in_memory = min( max_images_in_memory, num_output_images )
print 'Loading', images_in_memory, 'out of', num_output_images, 'into memory at once.'
if max_images_in_memory < 1:
    print >> sys.stderr, 'ERROR: Maximum RAM usage would be exceeded to save even one image',
    print >> sys.stderr, '(which would take', prod( output_shape ), 'bytes).'
    usage()

stack = empty( ( output_shape[0]*output_shape[1], images_in_memory, output_shape[2] ), dtype = uint8 )
count = 0
for mem_off in xrange( 0, num_output_images, images_in_memory ):
    mem_end = min( mem_off + images_in_memory, num_output_images )
    
    for frame_count, fname in enumerate( filenames ):
        print 'Loading "%s"' % ( fname, )
        img = Image.open( fname ).convert( 'RGB' )
        arr = asarray( img, dtype = uint8 )
        #arr = arr.reshape( -1, arr.shape[2] )[ mem_off : mem_end ]
        stack[ frame_count*image_shape[0] : frame_count*image_shape[0] + arr.shape[0], :mem_end-mem_off, : ] = arr[:,mem_off:mem_end,:]
    
    for save_off in xrange( 0, mem_end-mem_off ):
        outname = outname_basename + '-%04d.png' % ( count, )
        Image.fromarray( stack[ :, save_off, : ].reshape( output_shape ) ).save( outname )
        print 'Saved', outname
        
        count += 1
