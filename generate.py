import numpy as np
import skimage.transform as skt
import astra

def genphantom(sz=128):
    x = np.zeros((sz,sz), dtype=np.float32)
    x[sz//4:3*sz//4,sz//4:3*sz//4] = 1
    x = skt.rotate(x, np.random.random()*360)
    x[23*sz//48:25*sz//48,sz//2:] = 0
    return skt.rotate(x, np.random.random()*360)

pg = astra.create_proj_geom('parallel', 1, 256, np.linspace(0, np.pi, 3, False))
vg = astra.create_vol_geom(128)
pid = astra.create_projector('cuda',pg,vg)
w = astra.OpTomo(pid)

def gentraindata():
    x = genphantom()
    sino = w*x
    currec = w.reconstruct('SIRT_CUDA', sino, 100)
    ran_ang = np.random.random(2)*np.pi
    pg2 = astra.create_proj_geom('parallel', 1, 256, ran_ang)
    pid2 = astra.create_projector('cuda',pg2, vg)
    w2 = astra.OpTomo(pid2)
    gt = w2*x
    cur = w2*currec
    astra.projector.delete(pid2)
    inp = np.zeros((2,128,128), dtype=np.float32)
    tar = np.zeros((1, 128, 128), dtype=np.float32)
    inp[0] = currec
    inp[1] = (ran_ang[0]-np.pi/2)/np.pi
    tar[0] = (np.sqrt(((gt[:256] - cur[:256])**2).mean()) - 3.938124)/2.4107606
    return inp, tar

import os
import tqdm
import tifffile
os.makedirs('data', exist_ok=True)

for i in tqdm.trange(10000):
    inp, tar = gentraindata()
    tifffile.imsave('data/inp{:06d}.tiff'.format(i), inp)
    tifffile.imsave('data/tar{:06d}.tiff'.format(i), tar)