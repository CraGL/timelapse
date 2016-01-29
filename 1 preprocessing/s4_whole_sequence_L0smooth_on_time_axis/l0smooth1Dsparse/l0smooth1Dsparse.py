#!/usr/bin/env python

from numpy import *
from ptsv import ptsv

kUseCython = True
if kUseCython:
    import pyximport
    pyximport.install( setup_args={ 'script_args': ['--compiler=mingw32'], 'include_dirs': get_include() } )
    from mask_h import mask_h_threechannel

class Struct( object ): pass

def compute_h( S, num_channels, Lambda, beta, prepared ):
    assert num_channels >= 1
    assert len( S.shape ) == 2
    assert S.shape[1] % num_channels == 0
    
    h = prepared.h
    
    ## Simple:
    # h[:] = S[1:] - S[:-1]
    h[:] = S[1:]
    h[:] -= S[:-1]
    
    if kUseCython and num_channels == 3:
        mask_h_threechannel( h, Lambda/beta, prepared.maskscratch )
        return h
    
    h2 = prepared.h2
    h2sum = prepared.h2sum
    mask = prepared.mask
    
    ## Simple:
    # h[ (h**2).sum(-1) < Lambda/beta ] = 0.
    ## Treat each channel separately.
    # h2[:] = h**2
    square( h, out = h2 )
    
    h2sum[:] = 0
    for c in xrange( num_channels ):
        h2sum += h2[:,c::num_channels]
    del h2
    
    ## Set masked values of h to zero.
    # mask = mask < Lambda/beta
    # mask = repeat( mask, num_channels, axis = -1 )
    # h[ mask ] = 0.
    
    ## This is a little faster:
    # masked = empty( h.shape, dtype = bool )
    # masked[:,::num_channels] = mask < Lambda/beta
    # del mask
    # for c in xrange( 1, num_channels ):
    #     masked[:,c::num_channels] = masked[...,::num_channels]
    # h[ masked ] = 0.
    
    ## This is faster still:
    mask[:] = h2sum < Lambda/beta
    for c in xrange( num_channels ):
        h[:,c::num_channels][ mask ] = 0.
    
    return h

def prepare_solve_for_S( S, num_channels ):
    assert num_channels >= 1
    assert len( S.shape ) == 2
    assert S.shape[1] % num_channels == 0
    
    N = len(S)
    
    result = Struct()
    
    ## compute_h()
    h = empty( ( S.shape[0]-1, S.shape[1] ), order = 'F' )
    result.h = h
    if num_channels == 3 and kUseCython:
        result.maskscratch = empty( h.shape[0], order = 'F' )
    else:
        h2 = empty( h.shape, order = 'F' )
        h2sum = empty( ( h.shape[0], h.shape[1]//num_channels ), order = 'F' )
        mask = empty( h2sum.shape, dtype = bool, order = 'F' )
        
        result.h2 = h2
        result.h2sum = h2sum
        result.mask = mask
    
    ## solve_for_S()
    result.system_diag = empty( (N,1), order = 'F' )
    result.system_offdiag = empty( (N-1,1), order = 'F' )
    
    return result

def solve_for_S( S, I, h, beta, prepared, constrained = None, w_lsq = None ):
    ### WARNING: This function replaces the memory inside S with the return value.
    
    assert S.shape == I.shape
    
    N = len(S)
    
    # system = beta * gradTgrad + scipy.sparse.eye(N)
    system_diag = prepared.system_diag
    system_offdiag = prepared.system_offdiag
    system_offdiag[:] = -beta
    system_diag[:] = 2.*beta + 1.
    system_diag[ 0 ] = beta + 1.
    system_diag[ -1 ] = beta + 1.
    
    ## Simpler:
    # rhs = I + beta * ( grad.T * h )
    ## Faster:
    # rhs = (beta*grad.T) * h
    # rhs += I
    ## Faster still:
    rhs = S
    rhs[0,    :] = 0.
    rhs[1:,   :] = h
    rhs[ :-1, :] -= h
    rhs *= beta
    rhs += I
    
    #import pdb
    #pdb.set_trace()
    ## scipy.sparse.spdiags( system, ( -1, 0, 1 ), N, N )
    
    if constrained is not None and w_lsq != 0.:
        if w_lsq is None: w_lsq = 100.
        
        constrained = list( constrained )
        
        if w_lsq == 'hard':
            ## Zero the constrained rows and columns (update the RHS for the column),
            ## and set the diagonal to 1.
            
            ## Zero the constrained rows/columns.
            for c in constrained:
                if c > 0: rhs[ c-1,: ] -= system_offdiag[ c-1 ] * I[ c, : ]
                if c+1 < N: rhs[ c+1,: ] -= system_offdiag[ c ] * I[ c, : ]
                
                rhs[ c,: ] = I[ c,: ]
                
                system_diag[ c ] = 1.
                if c < N-1: system_offdiag[ c ] = 0.
                if c > 0: system_offdiag[ c-1 ] = 0.
        
        else:
            ## There is no += :-(
            system_diag[ constrained ] += w_lsq
            rhs[ constrained, : ] += w_lsq*I[ constrained, : ]
    
    ## lapack's ptsv directly:
    result = ptsv( system_diag, system_offdiag, rhs )
    assert may_share_memory( rhs, result )
    ## rhs is Snext
    return rhs

## Default parameters following the matlab code
kDefaultLambda = 2e-2
kDefaultKappa = 1.5
def l0smooth1D( I, axis = 0, channels = False, Lambda = None, kappa = None, harmonic = False, constrained = None, w_lsq = None ):
    '''
    The following documentation on parameters is taken from the matlab implentation
    provided with the paper "Image Smoothing via L0 Gradient Minimization" by
    Li Xu, Cewu Lu, Yi Xu, Jiaya Jia (SIGGRAPH Asia 2011):
    
    L0Smooth - Image Smoothing via L0 Gradient Minimization
       S = L0Smooth(Im, Lambda, kappa) performs L0 graidient smoothing of input
       image Im, with smoothness weight Lambda and rate kappa.
    
       Paras: 
       @Im    : Input UINT8 image, both grayscale and color images are acceptable.
       @Lambda: Smoothing parameter controlling the degree of smooth. (See [1]) 
                Typically it is within the range [1e-3, 1e-1], 2e-2 by default.
       @kappa : Parameter that controls the rate. (See [1])
                Small kappa results in more iteratioins and with sharper edges.   
                We select kappa in (1, 2].    
                kappa = 2 is suggested for natural images.  
    
       Example
       ==========
       Im  = imread('pflower.jpg');
       S  = L0Smooth(Im); # Default Parameters (Lambda = 2e-2, kappa = 2)
       figure, imshow(Im), figure, imshow(S);
    
    I added additional parameters:
        axis    : an integer specifying which axis of I is the 1D axis along which to smooth.
        channels: a boolean specifying whether-or-not the last axis of I is taken to be
                  the channels of a multichannel image. If channels is True, axis cannot
                  refer to the last axis.
        harmonic: a boolean specifying whether the l0 sparsity should be on the
                  color gradient or on the color laplacian. A sparse color gradient
                  is an image with constant color regions. A sparse color laplacian
                  is an image with linear color gradients.
        constrained: a list of indices along axis which will be constrained to remain
                     the same.
        w_lsq   : a floating point value (default: 100) specifying how strongly the
                  constrain on the constrained indices should be enforced.
    '''
    
    ## Default parameters following the matlab code
    if Lambda is None: Lambda = kDefaultLambda
    if kappa is None: kappa = kDefaultKappa
    beta0 = 2*Lambda
    betamax = 1e5
    
    I = asfarray(I).copy( order = 'F' )
    
    assert not channels or ( axis != -1 and axis < len( I.shape )-1 )
    num_channels = 1
    if channels:
        num_channels = I.shape[-1]
    
    if axis != 0:
        I = swapaxes( I, 0, axis )
    I_shape = I.shape
    I = I.reshape( I.shape[0], -1 )
    
    S = I.copy( order = 'F' )
    beta = beta0
    i = 0
    prepared = prepare_solve_for_S( S, num_channels )
    while beta < betamax:
        if i % 10 == 0:
            print beta, betamax
        
        h = compute_h( S, num_channels, Lambda, beta, prepared )
        S = solve_for_S( S, I, h, beta, prepared, constrained = constrained, w_lsq = w_lsq )
        
        i += 1
        beta *= kappa
    
    S = S.reshape( I_shape )
    if axis != 0:
        S = swapaxes( S, 0, axis )
    return S

def smooth_and_save( inpath, axis = 0, outpath = None, Lambda = None, kappa = None, w_lsq = None, constrained = None ):
    
    if outpath is None:
        import os
        inpath_base = os.path.splitext( inpath )[0]
        outpath = inpath_base + "-l0-axis_%d-Lambda_%3g-kappa_%g.png" % ( axis, kDefaultLambda if Lambda is None else Lambda, kDefaultKappa if kappa is None else kappa )
    
    import skimage.io
    print 'Loading:', inpath
    I = skimage.img_as_float( skimage.io.imread( inpath ) )
    assert len( I.shape ) == 3
    
    if w_lsq is not None and constrained is None:
        constrained = [ 0, I.shape[ axis ]-1 ]
    
    print 'Lambda:', Lambda
    print 'kappa:', kappa
    print 'Constraint weight:', w_lsq
    print 'Constraints:', constrained
    
    S = l0smooth1D( I, axis = axis, channels = True, Lambda = Lambda, kappa = kappa, constrained = constrained, w_lsq = w_lsq )
    S = S.clip(0,1)
    assert len( S.shape ) == 3
    
    skimage.io.imsave( outpath, S )
    print 'Saved:', outpath

def main():
    import sys
    argv = list( sys.argv[1:] )
    
    def usage():
        print >> sys.stderr, "Usage:", sys.argv[0], "[--lambda 2e-2] [--kappa 1.5] [--constraint-weight 0.] [--constraints [frame index, frame index, ...] (default: [0,num_frames-1])] path/to/input axis path/to/output"
        ## We liked Lambda = 5e-3, Kappa = 1.1
        sys.exit(-1)
    
    if len( argv ) == 0: usage()
    
    Lambda = None
    try:
        i = argv.index( '--lambda' )
        Lambda = float( argv[ i + 1 ] )
        del argv[ i : i + 2 ]
    except IndexError: usage()
    except ValueError: pass
    
    kappa = None
    try:
        i = argv.index( '--kappa' )
        kappa = float( argv[ i + 1 ] )
        del argv[ i : i + 2 ]
    except IndexError: usage()
    except ValueError: pass
    
    w_lsq = None
    try:
        i = argv.index( '--constraint-weight' )
        w_lsq = argv[ i + 1 ]
        if w_lsq != 'hard': w_lsq = float( w_lsq )
        del argv[ i : i + 2 ]
    except IndexError: usage()
    except ValueError: pass
    
    constrained = None
    try:
        i = argv.index( '--constraints' )
        import json
        constrained = list( json.loads( argv[ i + 1 ] ) )
        del argv[ i : i + 2 ]
    except IndexError: usage()
    except ValueError: pass
    
    inpath = argv.pop(0)
    axis = int( argv.pop(0) )
    outpath = argv.pop(0)
    
    import os
    if os.path.exists( outpath ):
        print >> sys.stderr, 'ERROR: Refusing to clobber output path:', outpath
        usage()
    if not os.path.isfile( inpath ):
        print >> sys.stderr, 'ERROR: Input path is not a file:', inpath
        usage()
    
    if len( argv ) != 0: usage()
    
    smooth_and_save( inpath, axis = axis, outpath = outpath, Lambda = Lambda, kappa = kappa, w_lsq = w_lsq, constrained = constrained )

if __name__ == '__main__': main()
