import ctypes.util
import numpy
import os, sys

def platform_shared_library_suffix():
    import sys
    result = 'so'
    if 'win' in sys.platform.lower(): result = 'dll'
    ## No else if, because we want darwin to override win (which is a substring of darwin)
    if 'darwin' in sys.platform.lower(): result = 'dylib'
    return result

'''
lib = None
libdirs = [ os.path.join( 'usr', 'lib' ) ]
for libdir in libdirs:
    libpath = os.path.join( libdir, 'liblapack.' + platform_shared_library_suffix() )
    if not os.path.exists( libpath ): continue
    
    lib = ctypes.cdll.LoadLibrary( libpath )
if lib is None:
    raise ImportError( 'Unable to find lapack' )
'''

lib = ctypes.util.find_library( 'lapack' )
if lib is None:
    ## Search for it right here:
    lib = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'liblapack.' + platform_shared_library_suffix() )
    if not os.path.isfile( lib ): lib = None
if lib is None:
    raise ImportError( 'Unable to find lapack' )
lib = ctypes.cdll.LoadLibrary( lib )

_dptsv = lib.dptsv_
_dptsv.restype = None
_dptsv.argtypes = \
    [
        # N
        ctypes.POINTER( ctypes.c_int ),
        # NRHS
        ctypes.POINTER( ctypes.c_int ),
        # D
        ctypes.POINTER( ctypes.c_double ),
        # E
        ctypes.POINTER( ctypes.c_double ),
        # B
        ctypes.POINTER( ctypes.c_double ),
        # LDB
        ctypes.POINTER( ctypes.c_int ),
        # INFO
        ctypes.POINTER( ctypes.c_int )
    ]
def ptsv( D, E, B ):
    assert len(D) == B.shape[0]
    assert len(E)+1 >= len(D)
    
    D = numpy.require( D, dtype = ctypes.c_double, requirements = [ 'F_CONTIGUOUS', 'WRITEABLE' ] )
    E = numpy.require( E, dtype = ctypes.c_double, requirements = [ 'F_CONTIGUOUS', 'WRITEABLE' ] )
    B = numpy.require( B, dtype = ctypes.c_double, requirements = [ 'F_CONTIGUOUS', 'WRITEABLE' ] )
    
    N = ctypes.c_int( len(D) )
    NRHS = ctypes.c_int( B.shape[1] )
    LDB = ctypes.c_int( max( 1, B.shape[0] ) )
    INFO = ctypes.c_int( 0 )
    
    _dptsv(
        ctypes.byref( N ),
        ctypes.byref( NRHS ),
        
        D.ctypes.data_as( ctypes.POINTER( ctypes.c_double ) ),
        E.ctypes.data_as( ctypes.POINTER( ctypes.c_double ) ),
        B.ctypes.data_as( ctypes.POINTER( ctypes.c_double ) ),
        
        ctypes.byref( LDB ),
        ctypes.byref( INFO )
        )
    
    if INFO.value < 0:
        raise RuntimeError( 'dptsv: The %d-th argument had an illegal value' % (-INFO.value) )
    elif INFO.value > 0:
        raise RuntimeError( 'dptsv: The leading minor of order %d is not positive definite; the solution has not been computed.' % INFO.value )
    
    return B
