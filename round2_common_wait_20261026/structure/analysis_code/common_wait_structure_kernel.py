"""Existing audited structure/contact kernels; explicit new-run destinations only.
Original source SHA256: 26789ad4636a3059eb80791ea1f8f5ab13c146c0e0123ebbc136e06b5d538d30
Only engine/potential paths are parameterized; scientific kernels unchanged.
"""
import json,hashlib,re,subprocess,os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
CONV=1.602176634
CACHE=None
POTENTIAL=None
ENGINE=None
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def tree_for(metal,bounds):
    xyz=metal[:,2:5].copy();xyz[:,0]+=1000;xyz[:,2]+=1000;xyz[:,1]=(xyz[:,1]-bounds[1,0])%(bounds[1,1]-bounds[1,0])
    return cKDTree(xyz,boxsize=[1e6,bounds[1,1]-bounds[1,0],1e6])

def bulk_graph(metal,bounds,cutoff=3.375):
    tree=tree_for(metal,bounds);pairs=tree.query_pairs(cutoff,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs),dtype=np.uint8),(pairs[:,0],pairs[:,1])),shape=(len(metal),len(metal))).tocsr()
    n,labels=connected_components(graph,directed=False)
    counts=np.bincount(labels);bulk=int(counts.argmax());mask=labels==bulk
    bottom=metal[:,4]<metal[:,4].min()+4
    assert mask[bottom].mean()>.99
    return mask,dict(cutoff_A=cutoff,components=int(n),bulk_atoms=int(mask.sum()),bottom_in_bulk_fraction=float(mask[bottom].mean()))

def pair_contact(atoms,bounds,bulk):
    tool=atoms[atoms[:,1]==5,2:5];metal=atoms[atoms[:,1]!=5];metal=metal[np.argsort(metal[:,0])];centre=tool.mean(axis=0)
    assert tool[:,1].min()-7.164>bounds[1,0] and tool[:,1].max()+7.164<bounds[1,1]
    indices=np.where(np.linalg.norm(metal[:,2:5]-centre,axis=1)<38.)[0]
    near=metal[indices];tree=cKDTree(tool);js=tree.query_ball_point(near[:,2:5],7.164)
    force=np.zeros(3);rep_force=np.zeros(3);attr_force=np.zeros(3);cut=np.zeros(len(metal),bool);rep=cut.copy();attr=cut.copy();pairs=0
    for atom,index,neighbours in zip(near,indices,js):
        if not neighbours:continue
        eps,sigma,rc=(.05477,2.168,5.4) if int(atom[1]) in (1,3) else (.13,2.86575,7.164)
        delta=tool[neighbours]-atom[2:5];r2=(delta*delta).sum(axis=1);keep=r2<rc**2;delta=delta[keep];r2=r2[keep]
        if not len(r2):continue
        q=(sigma*sigma/r2)**3;f=(24*eps/r2*(2*q*q-q))[:,None]*delta;rr=r2<2**(1/3)*sigma*sigma
        force+=f.sum(axis=0);rep_force+=f[rr].sum(axis=0);attr_force+=f[~rr].sum(axis=0);pairs+=len(r2)
        cut[index]=True;rep[index]=rr.any();attr[index]=(~rr).any()
    return dict(tool_centroid_A=centre.tolist(),cutoff_pairs=pairs,cutoff_metal_atoms=int(cut.sum()),repulsive_metal_atoms=int(rep.sum()),
                repulsive_bulk_atoms=int((rep&bulk).sum()),repulsive_nonbulk_atoms=int((rep&~bulk).sum()),attractive_bulk_atoms=int((attr&bulk).sum()),
                force_nN=(force*CONV).tolist(),repulsive_force_nN=(rep_force*CONV).tolist(),attractive_force_nN=(attr_force*CONV).tolist())

def write_dump(path,atoms,bounds,step):
    with path.open('w') as f:
        f.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(atoms)}\nITEM: BOX BOUNDS ss pp ss\n')
        for lo,hi in bounds:f.write(f'{lo:.17g} {hi:.17g}\n')
        f.write('ITEM: ATOMS id type x y z\n');np.savetxt(f,atoms,fmt=['%d','%d','%.17g','%.17g','%.17g'])

def engine_check(case,step,atoms,bounds,contact):
    base=CACHE/f'engine_{case}_{step}';directory=base;attempt=1
    while directory.exists():
        attempt+=1;directory=base.with_name(base.name+f'_attempt_{attempt:02d}')
    directory.mkdir()
    data=directory/'coordinates.data'
    with data.open('w') as f:
        f.write(f'Coordinate-only contact verification\n\n{len(atoms)} atoms\n5 atom types\n\n')
        verification_bounds=bounds.copy()
        for axis in (0,2):
            verification_bounds[axis,0]=min(bounds[axis,0],atoms[:,axis+2].min()-.01)
            verification_bounds[axis,1]=max(bounds[axis,1],atoms[:,axis+2].max()+.01)
        for label,(lo,hi) in zip('xyz',verification_bounds):f.write(f'{lo:.17g} {hi:.17g} {label}lo {label}hi\n')
        f.write('\nMasses\n\n1 63.546\n2 180.94788\n3 63.546\n4 180.94788\n5 12.01\n\nAtoms # atomic\n\n')
        np.savetxt(f,atoms,fmt=['%d','%d','%.17g','%.17g','%.17g'])
    potential=POTENTIAL;deck=directory/'verify.lmp'
    deck.write_text(f'''units metal
atom_style atomic
boundary s p s
read_data "{data}"
pair_style hybrid adp lj/cut 7.5
pair_coeff * * adp "{potential}" Cu Ta Cu Ta NULL
pair_coeff 1 5 lj/cut .05477 2.168 5.4
pair_coeff 3 5 lj/cut .05477 2.168 5.4
pair_coeff 2 5 lj/cut .13 2.86575 7.164
pair_coeff 4 5 lj/cut .13 2.86575 7.164
pair_coeff 5 5 none
neighbor 2 bin
group tool type 5
compute reaction tool reduce sum fx fy fz
thermo_style custom step c_reaction[1] c_reaction[2] c_reaction[3]
run 0
print "CONTACT_VERIFY $(c_reaction[1]:%.17g) $(c_reaction[2]:%.17g) $(c_reaction[3]:%.17g)"
''')
    engine=ENGINE;log=directory/'engine.log'
    with log.open('w') as out:
        proc=subprocess.run([str(engine),'-in',str(deck),'-log','none'],stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1'),timeout=180)
    assert proc.returncode==0,(case,log)
    match=re.search(r'^CONTACT_VERIFY ([^\n]+)',log.read_text(),re.M);assert match
    force=np.array(list(map(float,match[1].split())))*CONV;difference=force-np.array(contact['force_nN'])
    assert np.max(np.abs(difference))<1e-6,(case,force,contact['force_nN'])
    return dict(engine_sha256=digest(engine),input_sha256=digest(deck),log_sha256=digest(log),integration_steps=0,
                force_nN=force.tolist(),max_abs_difference_nN=float(np.abs(difference).max()),potential_sha256=digest(potential))

def height(xyz):
    ix=np.floor((xyz[:,0]-280)/5).astype(int);iy=np.floor((xyz[:,1]-28)/5).astype(int)
    good=(ix>=0)&(ix<36)&(iy>=0)&(iy<20)
    z=np.full((36,20),-np.inf);np.maximum.at(z,(ix[good],iy[good]),xyz[good,2]);z[~np.isfinite(z)]=np.nan
    return z

def remove_tool(frame,data):
    data.particles_.delete_elements(np.asarray(data.particles['Particle Type'])==5)
    data.cell_.pbc=(False,True,False)
