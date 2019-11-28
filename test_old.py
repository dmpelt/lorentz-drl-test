import msdnet

n = msdnet.network.NumberMSDNet.from_file('regr_params.h5', gpu=True)

import generate
import numpy as np

import astra
pg = astra.create_proj_geom('parallel', 1, 256, np.linspace(0, np.pi, 3, False))
vg = astra.create_vol_geom(128)
pid = astra.create_projector('cuda',pg,vg)
w = astra.OpTomo(pid)

x = generate.genphantom()

sino = w*x

currec = w.reconstruct('SIRT_CUDA', sino, 100)

angles = np.linspace(0,np.pi, 1024)
pg = astra.create_proj_geom('parallel', 1, 256, angles)
pid = astra.create_projector('cuda',pg,vg)
w = astra.OpTomo(pid)

gt = (w*x).reshape(w.sshape)
cs = (w*currec).reshape(w.sshape)
gtvals = np.sqrt(((gt - cs)**2).mean(1))

print(gtvals.shape)

inp_angles = (angles-np.pi/2)/np.pi

out = np.zeros(1024)

for i in range(len(angles)):
    inp = np.zeros((2, 128, 128), dtype=np.float32)
    inp[0] = currec
    inp[1] = inp_angles[i]
    out[i] = n.forward(inp)[0,0,0]

out *= 2.4107606
out += 3.938124


import pylab as pl

pl.plot(angles, out, label='network output')
pl.plot(angles, gtvals, label='ground truth')
pl.legend()

pl.show()
input()