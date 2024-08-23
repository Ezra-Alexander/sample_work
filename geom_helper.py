# from qd_helper import *
import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import random

def get_ind_uc(atoms,connectivity,inds):
    '''
    generates a set of boolean numpy arrays indicating which subset of entered arrays are under-coordinated
    Inputs:
        atoms: np array of element strs, NAtoms
        connectivity: np array of graph connectivities (ints), NAtoms x Coordination#
        inds: tuple of boolean np arrays to iterate through. X x NAtoms
    Outputs:
        uc_inds_final: tuple of boolean np arrays, each a subset of one in inds. X x NAtoms
    '''
    uc_inds=()
    for x in inds:
        uc_inds=uc_inds+([],)
    
    for i,atom in enumerate(atoms):
        for j in range(len(inds)):
            if inds[j][i] and len(connectivity[i])<4:
                uc_inds[j].append(True)
            else:
                uc_inds[j].append(False)

    uc_inds_final=()
    for i in range(len(uc_inds)):
        uc_inds_final=uc_inds_final+(np.array(uc_inds[i]),)

    return uc_inds_final


def get_angle_from_points(coord1,coord2,coord3):
    '''
    Like get_angle, but takes in 3 cartesian points instead of a full coordinates and distances array
    angle centered at coord1
    Inputs:
        coord1,2,3: len 3 np arrays of cartesian coordiantes (floats)
    Outputs:
        Angle: float. the angle, in degrees, between the 3 atoms, centered at coord1
    '''
    dist1=np.linalg.norm(coord1-coord2)
    dist2=np.linalg.norm(coord1-coord3)
    dist3=np.linalg.norm(coord2-coord3)
    num = dist1**2 + dist2**2 - dist3**2
    denom = 2*dist1*dist2
    angle = math.acos(num/denom)

    return math.degrees(angle)

def get_angle(coords,dists,index1,index2,index3):
    '''
    Computes a single bond angle, centered at index1, to index2 and index3, using xyz coords
    Inputs:
        coords: np array of cartesian coordinates (floats), NAtoms x3
        dists: np array of inter-atom distances (floats), NAtoms x NAtoms
        index1,2,3: ints that represent the 0-based indices of 3 atoms in the coordinate array
    Outputs:
        angle: float. the angle, in degrees, between the 3 atoms, centered at index 1
    '''
    dist1 = dists[index1][index2]
    dist2 = dists[index1][index3]
    dist3 = dists[index2][index3]
    num = dist1**2 + dist2**2 - dist3**2
    denom = 2*dist1*dist2
    angle = math.acos(num/denom)

    return math.degrees(angle)


def average_bond_lengths(atoms,dists,connectivity):
    '''
    for each bound pair of elements, computes an average bond length
    Inputs:
        atoms: np array of element strs, NAtoms
        dists: np array of inter-atom distances (floats), NAtoms x NAtoms
        connectivity: np array of graph connectivities, NAtoms x Coordination#
    Outputs:
        bond_avs: dictionary of average bond lengths, max NElements x NElements
    '''
    bond_sums={}
    bond_counts={}
    for i,atom in enumerate(atoms):
        for bond in connectivity[i]:
            if (atom not in bond_sums): 
                bond_sums[atom]={atoms[bond]:dists[i][bond]}
                bond_counts[atom]={atoms[bond]:1}
            elif (atoms[bond] not in bond_sums[atom]):
                bond_sums[atom][atoms[bond]]=dists[i][bond]
                bond_counts[atom][atoms[bond]]=1
            else:
                bond_sums[atom][atoms[bond]]=bond_sums[atom][atoms[bond]]+dists[i][bond]
                bond_counts[atom][atoms[bond]]=bond_counts[atom][atoms[bond]]+1

    bond_avs=bond_sums.copy()
    for element in bond_sums:
        for j in bond_sums[element]:
            bond_avs[element][j]=bond_sums[element][j]/bond_counts[element][j]
    #print(bond_avs)

    #add missing values as just the sum of the covalent radii
    covalent_radii={"In":1.42,"Ga":1.22,"P":1.07,"Cl":1.02,"F":0.57,"Zn":1.22,"S":1.05,"Se":1.20,"O":0.66,"Al":1.21,"H":0.31,"Br":1.2,"Si":1.11,"C":0.76,"N":0.71,"Li":1.28}
    for element in bond_avs:
        for element2 in bond_avs:
            if element2 not in bond_avs[element]:
                bond_avs[element][element2]=covalent_radii[element]+covalent_radii[element2]

    return bond_avs




def vector_projector(to_project,reference1,reference2):
    '''
    creates a new basis set of: reference 1, the orthogonal to reference 1 that is coplanar and codirectional with reference 2, and their cross product
    then projects the to_project vector into this new basis set
    The references are normalized by this function
    the projected vector is not normalized
    Inputs:
        to_project: np array of cartesian coordinates, len 3
        reference1: np array of cartesian coordinates, len 3
        reference2: np array of cartesian coordinates, len 3
    Outputs:
        projected: np array of cartesian coordinates, len 3
    '''

    unit1=reference1/np.linalg.norm(reference1)
    basis2=unit1-(reference2/np.dot(reference2,unit1))
    unit2=basis2/np.linalg.norm(basis2)
    basis3=np.cross(unit1,unit2)
    unit3=basis3/np.linalg.norm(basis3)
    projected=np.matmul(np.linalg.inv(np.column_stack((unit1,unit2,unit3))),to_project)

    return projected

def center_coords(coords):
    '''
    Centers a set of cartesian coordinates on their linear average
    Inputs:
        coords: np array of cartesian coordinates, NAtoms x 3
    Outputs:
        centered: np array of centered cartesian coordinates, NAtoms x 3
    '''
    center=np.mean(coords,axis=0)
    
    return coords-center


def average_over_element(atoms,property_list):
    '''
    A general function to take a list of properties for each atom in a structure 
    and compute the average value of that property for each elemental species

    Inputs:
        atoms: np array of elements, NAtoms
        property_list: np array of some numerical property, NAtoms
    Outputs:
        averages: dictionary in the form {'element':average, ...}
    '''
    averages={}
    counts={}
    for i,atom in enumerate(atoms):
        if atom not in averages:
            averages[atom]=0
            counts[atom]=0
        averages[atom]=averages[atom]+property_list[i]
        counts[atom]=counts[atom]+1
    for key in averages:
        averages[key]=averages[key]/counts[key]

    return averages


def approximate_dipole(atoms,coords):
    '''
    Uses a pre-established dictionary to approximate the dipole moment of an .xyz file if every atom is in its default oxidation state
    Coordinates are first shifted to be centered at the average of all coordinates
    Both the shifted coordinates and the dipole vector are returned

    Inputs:
        atoms: np array of elements, NAtoms
        coords: np array of cartesian coordinates per atom, NAtoms x 3
    Outputs:
        dipole: Dipole moment vector, 3
        shifted_coords: np array of cartesian coordinates shifted to their average, NAtoms x 3
    '''

    charge_dictionary={"In":3,"P":-3,"Ga":3,"F":-1,"Cl":-1,"Al":3,"Zn":2,"Cd":2,"S":-2,"O":-2,"Se":-2}
    mass_dictionary={"In":114.82,"P":30.974,"Ga":69.723,"F":18.998,"Cl":35.45,"Al":26.982,"Zn":65.38,"Cd":112.41,"S":32.06,"O":15.999,"Se":78.971}

    com=[0,0,0]
    for i,atom in enumerate(atoms):
        com = com + coords[i]#(mass_dictionary[atom]*coords[i])
    com = com / len(atoms)
    shifted_coords=coords-com

    dipole=[0,0,0]
    for i,atom in enumerate(atoms):
        dipole=dipole+(charge_dictionary[atom]*shifted_coords[i])

    #print(dipole, np.linalg.norm(dipole))
    return dipole, shifted_coords

def get_charge(atoms):
    '''
    Uses a pre-established dictionary to compute the total charge of a .xyz file if every atom is in its default oxidation state

    Inputs:
        atoms: np array of elements, NAtoms
    Outputs:
        charge: int
    '''
    charge_dictionary={"In":3,"P":-3,"Ga":3,"F":-1,"Cl":-1,"Al":3,"Zn":2,"Cd":2,"S":-2,"O":-2,"Se":-2}
    charge=0
    for atom in atoms:
        charge=charge+charge_dictionary[atom]
    return charge

def get_n_closest(coords,n,index):
    '''
    Returns a list of total indices of the n atoms that are closest to the given atom (including that atom)

    Inputs:
        coords: np array of xyz coordinates, Natomsx3
        n: the size of the desired subset. integer
        index: the total index of the atom in question. [0,NAtoms-1]
    Outputs:
        n_closest: list of total indices of closest atoms
    '''
    dists=dist_all_points(coords)
    sorted_indices=np.argsort(dists[index])
    return sorted_indices[:n]


def to_total_index(atoms,element,index):
    '''
    Returns the total index given an atom-specific index

    Inputs:
        atoms: np array of elements, NAtoms
        element: str, on the periodic table. appears in atoms
        index: the atom-specific index of the atom in question. [1,NElement]
    outputs:
        i: outputs the total index of the atom in question. [0,NAtoms-1]
    '''
    count=0
    for i,atom in enumerate(atoms):
        if atom==element:
            count=count+1
            if count==index:
                return i

    raise Exception("Element in question does not appear in this list of atoms")

def delete_atoms(coords,atoms,to_delete):
    '''
    Takes in a set of atoms and coords and returns those same atoms and coords minus a list of specified atoms

    Inputs:
        coords: npy array of xyz coordinates, Natomsx3
        atoms: npy array of elements, Natoms
        to_delete: list of total indices to exclude
    Outputs:
        new_coords: the same coordinates minus those with the specified index
        new_atoms: the same elements minus those with the specified index
    '''

    new_coords=[]
    new_atoms=[]
    for i,atom in enumerate(atoms):
        if i not in to_delete:
            new_coords.append(coords[i])
            new_atoms.append(atom)

    return np.array(new_coords), np.array(new_atoms)




def depth_first_search(connectivity,start,visited,stack):
    '''
    Perfrom a depth first search on one of my molecular connectivity graphs, starting from a chosen atom and finding all indirectly connected atoms
    recursive
    can be used to find only subbranches if your initial visited includes pathways you want to cut off

    Inputs:
        connectivity: list, NAtoms in length, where each entry is a list of the total 0-based indices of all atoms bound to that atom
        start: the 0-based total index of the atom to start from. int
        visited: list of visited atoms (again, 0-based total indices). initial call can include atoms bound to start along which to not explore
        stack: list of paths yet to explore. again, 0-based total indices
    Outputs:
        visited: the final version of the visited array after recursion is finished
        stack: at the end, it should be empty
    '''
    visited.append(start)
    for atom in connectivity[start]:
        if atom not in visited:
            stack.append(atom)

    while len(stack)>0:
        start=stack.pop()
        visited,stack=depth_first_search(connectivity,start,visited,stack)

    return visited,stack




def generate_dihedral_points(start,coords,atoms):
    '''
    From a starting atom, generate 3 additional points that are all sort of connected with each other

    Inputs:
        start: integer total atom index, 0-based, of atom in question
        coords: NAtomsx3 npy array of xyz coordinates per atom
        atoms: NAtoms npy array of atomic elements
    Outputs:
        dihedrals: a list of 3 total atom indices, 0 based, for 3 different non-start atoms that are graphically connected
    '''

    dists=dist_all_points(coords)
    connect=smart_connectivity_finder(dists,atoms)

    dihedrals=[]
    backtrack=0
    while len(dihedrals)<3:
        if len(dihedrals)==0:
            first=random.randint(0,len(connect[start])-1)
            last=connect[start][first]
            dihedrals.append(last)
            backtrack=0
        else:
            for k,bond in enumerate(connect[last]):
                if (bond not in dihedrals) and (bond != start):
                    last=bond
                    dihedrals.append(last)
                    backtrack=0
                    break
                if k+1==len(connect[last]):
                    if len(dihedrals)>backtrack:
                        backtrack=backtrack+1
                        last=dihedrals[-backtrack]
                    else:
                        last=start
    return dihedrals


def get_dihedral(p1,p2,p3,p4):
    '''
    This function takes in four xyz coordinates (length 3 np arrays) and outputs a dihedral angle (float)
    '''

    v1= p1-p2
    v2=p3-p2
    v3=p4-p3

    u1=v1/np.linalg.norm(v1)
    u2=v2/np.linalg.norm(v2)
    u3=v3/np.linalg.norm(v3)

    proj1=u1-(np.dot(u1,u2)*u2)
    proj2=u3-(np.dot(u3,u2)*u3)

    x=np.dot(proj1,proj2)
    y=np.dot(np.cross(u2,proj1),proj2)

    dihedral=np.degrees(np.arctan2(y,x))

    return dihedral

def get_adjacency_matrix(connect):
    '''
    This function takes one of my connectivity graphs and converts it to an adjacency matrix

    Inputs:
        connect: a np array of length NAtoms, where each element is an array of indices that element is bound to
    Outputs:
        adjmat: a np matrix of size NAtoms x Natoms, which is valued 1 if the two atoms are bound and 0 otherwise
    '''

    natoms=len(connect)
    adjmat=np.zeros((natoms,natoms))

    for i,atom in enumerate(connect):
        for bond in atom:
            adjmat[i][bond]=1
            adjmat[bond][i]=1

    return adjmat

def get_mulliken(out_file,atoms):
    '''
    writes the mulliken populations in a .out file to a numpy array
    if the file has multiple jobs, takes pops from the first one

    Inputs:
        out_file: a qchem.out file, as a string
        atoms: np array of elements, NAtoms long
    Outputs:
        mull_pops: np array of floats, NAtoms long
    '''

    with open(out_file,"r") as out:
        for i,line in enumerate(out):
            if line.find("Ground-State Mulliken Net Atomic Charges")!=-1:
                x=np.loadtxt(out_file,skiprows=(i+4),max_rows=len(atoms),usecols=2,encoding="ISO-8859-1")
                return x


def local_equivalence_test(atom1,atom2,atoms,connectivity):
    '''

    In these large nanocrystals where symmetry isn't really a factor
    We have this idea that certain atoms are locally the same
    If they have similar coordination environments

    This function determines if two given atoms in a nanocrystal are locally identical
    We consider the local environment up to second nearest neighbors
    Which is to say we consider the atomic species and coordination number of each atom within next nearest neighbors
    Whwere atoms in the second coordination shell are tagged by the element and neighbor they passed through
    
    Inputs:
        atom1/atom2: int, overall index of atom 1 / atom 2 (zero-based)
        atoms: np array of element labels, in order. NAtoms long
        connectivity: np array that forms a connectivity graph, based on bonding cutoffs. NAtoms x NAtoms
    Outputs:
        Boolean. True if the two atoms are locally equivalent
    '''

    if atoms[atom1]!=atoms[atom2]: #if the elements are different
        return False

    if len(connectivity[atom1])!=len(connectivity[atom2]): #if the coordination number is different
        return False

    atom1_bonds=[]
    for element in connectivity[atom1]:
        atom1_bonds.append(atoms[element])

    atom2_bonds=[]
    for element in connectivity[atom2]:
        atom2_bonds.append(atoms[element])

    if Counter(atom1_bonds) != Counter(atom2_bonds): #if the coordinated elements are different
        return False

    atom1_nn =[[atoms[x],len(connectivity[x])] for x in connectivity[atom1]]
    atom2_nn =[[atoms[x],len(connectivity[x])] for x in connectivity[atom2]]    

    atom1_nn=sorted(atom1_nn)
    atom2_nn=sorted(atom2_nn)

    if atom1_nn != atom2_nn: #if the coordinated coordination numbers are different
        return False


    atom1_2nn=[]
    for bond in connectivity[atom1]:
        for bond2 in connectivity[bond]:
            if bond2!=atom1:
                atom1_2nn.append([atoms[bond],len(connectivity[bond]),atoms[bond2],len(connectivity[bond2])])
    atom1_2nn=sorted(atom1_2nn)

    atom2_2nn=[]
    for bond in connectivity[atom2]:
        for bond2 in connectivity[bond]:
            if bond2!=atom2:
                atom2_2nn.append([atoms[bond],len(connectivity[bond]),atoms[bond2],len(connectivity[bond2])])
    atom2_2nn=sorted(atom2_2nn)

    if atom1_2nn!=atom2_2nn:
        return False

    return True

def get_close_angles(xyz,dists,connect):
    '''
    Like get all angles but the fast version is faster because you don't actually do the full 3d loop

    Inputs:
        xyz: np array of xyz coordinates for the QD. Has shape (Natoms,3)
        dists: np array of distances between all atoms (NAtoms x NAtoms)
        connect: np connectivity graph, NAtoms x N_bonds
    Outputs:
        angles: np array of angles. NAtoms x NAtoms x NAtoms. all nonbonded entries are NaN. first index is the center of the angle, so (a,b,c)!=(b,a,c)=(b,c,a)
    '''
    angles=np.zeros((len(xyz),len(xyz),len(xyz)))
    angles[:]=np.nan
    for i,center in enumerate(xyz):
        for j,atom1 in enumerate(connect[i]):
            for k,atom2 in enumerate(connect[i]):
                if atom1>atom2:
                    dist1 = dists[i][atom1]
                    dist2 = dists[i][atom2]
                    dist3 = dists[atom1][atom2]
                    num = dist1**2 + dist2**2 - dist3**2
                    denom = 2*dist1*dist2
                    angle = math.acos(num/denom)
                    angles[i,atom1,atom2]=math.degrees(angle)
                    angles[i,atom2,atom1]=math.degrees(angle)

    return angles

def get_all_angles(xyz,fast=False):
    '''

    Function that calculates all angles between all atoms in a .xyz
    No distances involves because it's cleaner this way
    any angle with repeated indices is set to 0

    Inputs:
        xyz: np array of xyz coordinates for the QD. Has shape (Natoms,3)
        fast: boolean. If true, skips computing angles between atoms > 4 A away
    Outputs: 
        angles: np array of angles. NAtoms x NAtoms x NAtoms. first index is the center of the angle, so (a,b,c)!=(b,a,c)=(b,c,a)
    '''
    dists=dist_all_points(xyz)

    if fast:
        angles=np.zeros((len(xyz),len(xyz),len(xyz)))
        angles[:]=np.nan
        for i,center in enumerate(xyz):
            for j,atom1 in enumerate(xyz):
                for k,atom2 in enumerate(xyz):
                    if i!=j and i!=k and j!=k:
                        dist1 = dists[i][j]
                        dist2 = dists[i][k]
                        dist3 = dists[j][k]
                        if dist1<4 and dist2<4 and dist3<4:
                            num = dist1**2 + dist2**2 - dist3**2
                            denom = 2*dist1*dist2
                            angle = math.acos(num/denom)
                            angles[i,j,k]=math.degrees(angle)
    else:
        angles=np.zeros((len(xyz),len(xyz),len(xyz)))
        for i,center in enumerate(xyz):
            #centers = []
            for j,atom1 in enumerate(xyz):
               #index_1=[]
                for k,atom2 in enumerate(xyz):
                    if i!=j and i!=k and j!=k:
                        dist1 = dists[i][j]
                        dist2 = dists[i][k]
                        dist3 = dists[j][k]
                        num = dist1**2 + dist2**2 - dist3**2
                        denom = 2*dist1*dist2
                        angle = math.acos(num/denom)
                        angles[i,j,k]=math.degrees(angle)
                        #index_1.append(math.degrees(angle))
                    else:
                        #index_1.append(np.NAN)
                        angles[i,j,k]=np.nan
                #centers.append(index_1)
            #angles.append(centers)



    #return np.array(angles)
    return angles


def get_geom_cp2k(xyz):
    '''
    Gets the last (converged) .xyz from a c2pk CP2K-pos-1.xyz

    Inputs:
        xyz: String. The name of the cp2k .xyz file
    Outputs:
        atoms: np array of atom names, in order. NAtoms long
        coords: np array of (x,y,z) coordinated in angstroms. NAtoms x 3
    '''

    f = open(xyz, "r")
    lines=f.readlines()
    n_atoms=int(lines[0].strip())
    n_iterations=len(lines)/(n_atoms+2)
    start_line=int(((n_atoms+2)*(n_iterations-1))+2) #0-indexed

    xyz_coords = np.loadtxt(xyz,skiprows=start_line,usecols=(1,2,3))
    atom_names = np.loadtxt(xyz,skiprows=start_line,usecols=(0,),dtype=str)
    return xyz_coords, atom_names


def smart_connectivity_finder(dists,atoms,flexibility=1.25):
    '''
    Like the original connectivity finder, but uses a dictionary to implement atom-specific bond thresholds (sum of covalent radii * 1.25)

    makes an NAtoms list containing, for each atom, the indices of all other atoms within cutoff of that atom

    Needs to use a lower threshold for homodimers (sum of covalent radii * 1.1)

    Inputs:
        dists: np array with distances between all atoms, size (Natoms, Natoms)
    Outputs:
        connect: NAtom alist of NBonded lists of python-adjusted indices for all other atoms within cutoff of that atom
    '''

    covalent_radii={"In":1.42,"Ga":1.22,"P":1.07,"Cl":1.02,"F":0.57,"Zn":1.22,"S":1.05,"Se":1.20,"O":0.66,"Al":1.21,"H":0.31,"Br":1.2,"Si":1.11,"C":0.76,"N":0.71,"Li":1.28}
    #bond_maxes={"In":{"F":2.49,"Cl":3.05,"P":3.11,"S":3.09,"Se":3.28},"Ga":{"F":2.24,"Cl":2.80,"P":2.86,"S":2.84,"Se":3.02},"P":{"O":2.16,"In":3.11,"Ga":2.86,"Zn":2.86},"F":{"In":2.49,"Ga":2.24,"Zn":2.24},"Cl":{"In":3.05,"Ga":2.80,"Zn":2.80},"O":{"P":2.16},"Zn":{"P":2.86,"F":2.24,"Cl":2.80,"S":2.84,"Se":3.02},"S":{"In":3.09,"Ga":2.84,"Zn":2.84},"Se":{"In":3.09,"Ga":3.02,"Zn":3.02}}

    connectivity=[]
    for i,atom in enumerate(dists):
        bonds=[]
        for j,dist in enumerate(atom):
            cutoff=(covalent_radii[atoms[i]]+covalent_radii[atoms[j]])*flexibility
            if atoms[i]==atoms[j] or (atoms[i]=="In" and atoms[j]=="Ga") or (atoms[j]=="In" and atoms[i]=="Ga"):
                cutoff=(covalent_radii[atoms[i]]+covalent_radii[atoms[j]])*1.1
            if float(dist) < cutoff and i!=j:
                bonds.append(j)
        connectivity.append(bonds)

    return connectivity

def get_coordination_sphere(center,radius,npoints):
    '''
    Meant to generate a numpy array of xyz coordinates that lie along a given sphere
    For use in applications where the coordination sphere must be explored to decide the best placement of a ligand, for example
    Generates points in spherical coordinates, then converts to cartesian

    Inputs:
        center: 3x1 np array of the form [X Y Z]
        radius: float. the bond length
        npoints: the final product contains npoints cartesian points. The more the merrier
    Outputs:
        sphere_points: np array of cartesian coordinates along the sphere. npoints x 3
    '''
    #randomly chosen, not uniform over an angle. Fight me 
    theta = np.random.uniform(0,2*np.pi,npoints)
    phi = np.random.uniform(0,np.pi,npoints)

    #convert to cartesian
    x = center[0] + (radius*np.sin(phi)*np.cos(theta))
    y = center[1] + (radius*np.sin(phi)*np.sin(theta))
    z = center[2] + (radius*np.cos(phi))

    sphere_points=np.array([x,y,z]).T

    return sphere_points



def bring_to_4c(atoms,coords,target_index, ligand,threshold):
    '''
    A function that passivated a given atom from as low as 1c up to four-coordinate in a tetrahedral geometry with a chosen ligand

    Inputs:
        atoms: np array of atom names, in order. NAtoms long
        coords: np array of atomic coords, in the same order. NAtoms x 3
        target_index: int. the overall index, python adjusted, of the atom you are targeting
        ligand: str. the element you will use to passivate the target. Note that bond lengths are hard-coded
        threshold: float. obsolete
    Outputs:
        temp_atoms: updated np array of atom names, in order. NAtoms long
        temp_coords: updated np array of atomic coords, in the same order. NAtoms x 3
    '''

    bond_lengths={"In":{"F":2.02,"Cl":2.37,"N":2.13},"Ga":{"F":1.79,"Cl":2.19,"N":1.93},"P":{"O":1.53}} #hard coded (terminal) bond lengths. not everything will be supported. Don't have to be perfect, these are just pre-opt estimates

    bond_length=bond_lengths[atoms[target_index]][ligand]

    temp_coords=np.copy(coords)
    temp_atoms=np.copy(atoms)

    dists=dist_all_points(temp_coords)
    connectivity=smart_connectivity_finder(dists,atoms)

    print(len(connectivity[target_index]))

    if len(connectivity[target_index])==1:
        #the hard part for a 1-coordinate target is choosing the direction of the new bond
        anchor=connectivity[target_index][0]
        anchor_bonds=connectivity[anchor]
        vector=temp_coords[anchor]-temp_coords[anchor_bonds[0]]
        unit=vector/np.linalg.norm(vector)
        added_coords=temp_coords[target_index]+(unit*bond_length)
        temp_coords=np.append(temp_coords,[added_coords],axis=0)
        temp_atoms=np.append(temp_atoms,ligand)

    dists=dist_all_points(temp_coords)
    connectivity=smart_connectivity_finder(dists,atoms)
    if len(connectivity[target_index])==2:
        v1=temp_coords[target_index]-temp_coords[connectivity[target_index][0]]
        v2=temp_coords[target_index]-temp_coords[connectivity[target_index][1]]

        parallel=(v1+v2)/np.linalg.norm(v1+v2)
        perp=np.cross(v1,v2)
        perp_unit=perp/np.linalg.norm(perp)

        adjacent=bond_length*math.cos(math.radians(109.5/2))
        opposite=bond_length*math.sin(math.radians(109.5/2))

        added_coords=temp_coords[target_index]+(parallel*adjacent)+(perp_unit*opposite)
        temp_coords=np.append(temp_coords,[added_coords],axis=0)
        temp_atoms=np.append(temp_atoms,ligand)


    dists=dist_all_points(temp_coords)
    connectivity=smart_connectivity_finder(dists,atoms)
    if len(connectivity[target_index])==3:
        temp_atoms,temp_coords=geom_adder(temp_atoms,temp_coords,target_index,ligand)

    return temp_atoms, temp_coords



def geom_adder(atoms,coords,target_index,add_element):
    '''
    From the script of the same name
    A script meant to add a chosen element to a given 3-coordinate target in the 4th tetrahedral position
    Technically also works for adding something to a 4+ coordinate target, but it ends up putting it somewhere weird usually

    THIS FUNCTION IS OUTDATED - USE MASS_ADDER INSTEAD

    Inputs:
        atoms: np array of atom names, in order. NAtoms long
        coords: np array of atomic coords, in the same order. NAtoms x 3
        target_index: int. the overall index, python adjusted, of the atom you are targeting
        add_element: str. the element you wish to add. note that target-add bond lengths are hard coded and not all may be supported yet
    '''
    #find vector along 4th dimension
    ind_In = atoms=='In'
    ind_P = atoms=='P'
    ind_Ga = atoms=="Ga"
    ind_InGa = np.logical_or(ind_In,ind_Ga)
    ind_cat=np.logical_or(ind_InGa,atoms=="Al")
    ind_InP= np.logical_or(ind_In,ind_P)
    ind_lig = (atoms == "Cl")

    all_dists,inp_dists,inf_dists,inpf_dists,pin_dists = get_dists(coords,ind_cat,ind_P,ind_lig)
    #all_dists,inp_dists,inf_dists,inpf_dists,pin_dists = get_dists(coords,ind_Ga,ind_P,ind_lig)

    target_dists = all_dists[target_index]
    #sorted_dists = np.sort(target_dists)
    indexes = [1000,1000,1000]
    dists = [1000,1000,1000]
    for i,dist in enumerate(target_dists):
        if dist < max(dists) and i!=target_index:
            indexes.pop(dists.index(max(dists)))
            dists.pop(dists.index(max(dists)))
            indexes.append(i)
            dists.append(dist)

    ind_1 = indexes[0]
    ind_2 = indexes[1]
    ind_3 = indexes[2]

    #bond_1 = sorted_dists[1]
    #bond_2 = sorted_dists[2]
    #bond_3 = sorted_dists[3]

    #this assumes that the three closest dists are not exactly equal
    #ind_1 = np.where(target_dists==bond_1)[0][0]
    #ind_2 = np.where(target_dists==bond_2)[0][0]
    #ind_3 = np.where(target_dists==bond_3)[0][0]

    # dummy = (coords[ind_1] + coords[ind_2] + coords[ind_3])/3

    #dummy = (coords[ind_2] + coords[ind_3])/3 #temp

    v1=coords[target_index]-coords[ind_1]
    v2=coords[target_index]-coords[ind_2]
    v3=coords[target_index]-coords[ind_3]

    u1=v1/np.linalg.norm(v1)
    u2=v2/np.linalg.norm(v2)
    u3=v3/np.linalg.norm(v3)

    vector=u1+u2+u3

    #vector = coords[target_index]-dummy
    mag = np.linalg.norm(vector)
    unit = vector/mag

    #Need to add bond lengths of each type as I go
    if add_element == "In" and atoms[target_index] == "P":
        new_coords = coords[target_index] + unit*2.58
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element == "F" and atoms[target_index] == "In":
        new_coords = coords[target_index] + unit*2.2
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element == "O" and atoms[target_index] == "In":
        new_coords = coords[target_index] + unit*2.05
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element == "Ga" and atoms[target_index] == "P":
        new_coords = coords[target_index] + unit*2.5
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="F" and atoms[target_index]=="Ga":
        new_coords = coords[target_index] + unit*2.12
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="Al" and atoms[target_index]=="P":
        new_coords = coords[target_index] + unit*2.32
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="Cl" and atoms[target_index]=="In":
        new_coords = coords[target_index] + unit*2.36
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="Cl" and atoms[target_index]=="Zn":
        new_coords = coords[target_index] + unit*2.186
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="Cl" and atoms[target_index]=="Al":
        new_coords = coords[target_index] + unit*2.116
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="Cl" and atoms[target_index]=="Ga":
        new_coords = coords[target_index] + unit*2.19
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="O" and atoms[target_index]=="P":
        new_coords = coords[target_index] + unit*1.53
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="P" and atoms[target_index]=="Ga":
        new_coords = coords[target_index] + unit*2.48
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="N" and atoms[target_index]=="In":
        new_coords = coords[target_index] + unit*2.13
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="N" and atoms[target_index]=="Ga":
        new_coords = coords[target_index] + unit*1.93
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    elif add_element=="P" and atoms[target_index]=="In":
        new_coords = coords[target_index] + unit*2.49
        print("Done!")
        new_atoms = np.append(atoms,add_element)
        final_coords = np.append(coords,[new_coords], axis=0)
    else:
        print("Warning! The bond length between the element you are adding and the element you are targeting is not yet supported!")
    return new_atoms,final_coords

def get_label(atoms,coords,target_element,target_index,ligand):
    '''
    Function that uses the connectivity mapping to label a given atom as being either under-coordinated, surface, bulk, etc.
    With delta-DFT in mind but there are probably other uses

    Inputs:
        atoms: np array of atom names, in order. NAtoms long
        coords: np array of atomic coords, in the same order. NAtoms x 3
        target_element: string. the element of the atom you are targeting. this could be written with overall indices but i prefer this
        target_index: int. the atom-specific index of the atom you are targeting
        ligand: the atom which we consider to be the ligand. if it isn't ligand or InGaP, it is a dopant
    Outputs:
        label: string. either "Under-Coordinated", "Bulk", "Bound to Trap", or "Bound to Dopant"
            Anything other than In, Ga, P, or ligand is considered a dopant
            Only bulk "near dopants" will be included
    '''
    dists=dist_all_points(coords)
    connectivity=smart_connectivity_finder(dists,atoms)
    count=0
    uc_flag=False
    dopant_flag=False
    shell_flag=False
    for i,atom in enumerate(atoms):
        if atom==target_element:
            count=count+1
            if count==target_index: #find your element

                if len(connectivity[i])<4: #test under-coordinated P
                    # for j,bond in enumerate(connectivity[i]):
                    #     if len(connectivity[bond])<4 and atoms[bond]!=ligand:
                    #         return "Bound to Trap"
                    return "Under-Coordinated"
                else:
                    #I want to add a function to make "near a trap but not bound to it" into "Bound to Trap"
                    # for j,dist in enumerate(dists[i]):
                    #     if dist<5 and (atoms[j]=="In" or atoms[j]=="Ga" or atoms[j]=="Al"): #arbitrary
                    #         if len(connectivity[j])<4:
                    #             #print(atoms[j],to_atom_specific_index(atoms,j))
                    #             uc_flag=True

                    for j,bond in enumerate(connectivity[i]):
                        if len(connectivity[bond])<4 and atoms[bond]!=ligand:
                            uc_flag=True
                            #return "Bound to Trap"
                        if atoms[bond]!=ligand and atoms[bond]!="In" and atoms[bond]!="Ga"  and atoms[bond]!="P" and atoms[bond]!="Al":
                            dopant_flag=True
                        # for k,bond2 in enumerate(connectivity[bond]):
                        #     if atoms[bond2]!=ligand and atoms[bond2]!="In" and atoms[bond2]!="Ga"  and atoms[bond2]!="P" and atoms[bond2]!="Al":
                        #         dopant_flag=True
                        # if atoms[bond]=="Al":
                        #     shell_flag=True
                            #return "Bound to Dopant"
                        #elif atoms[bond]==ligand:
                        #    return "Surface"
                        #else:
                        #    for k,bond2 in enumerate(connectivity[bond]):
                        #        if atoms[bond2]==ligand:
                        #            return "Surface"


                    if dopant_flag==True: #manage ordering
                        return "Bound to Dopant"         
                    if uc_flag==True:
                        return "Bound to Trap"
                    # if shell_flag==True:
                    #     return "Surface"
                               


                    return "Bulk"



def to_atom_specific_index(atoms,overall_index):
    '''
    Function that converts from overall indices (i.e. python index 239 is the 240th atom in the structure)
    to atom-specific indices (i.e. P 12, the 12th Phosphorus)

    Inputs:
        atoms: np array of atom names, in order. shape (Natoms)
        overall_index: positive integer, 0-based indexed.
    '''
    target_atom=atoms[overall_index]
    count=0
    for i,atom in enumerate(atoms):
        if atom==target_atom:
            count=count+1
            if i==overall_index:
                return count


def get_underc_index_variable(xyz,ind_Cd,ind_Se,ind_lig,lig_cutoff,p_cutoff,nncutoff):
    '''
    Function that finds undercoordinated Cd and Se atoms in a QD.

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        lig_cutoff: upper cutoff for In-F bonds
        p_cutoff: upper cutoff for In-P bonds
        nncutoff: number of nearest neighbors to be considered "fully coordinated"
                  (< this classified as "undercoordinated")
    '''
    all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyz,ind_Cd,ind_Se,ind_lig)

    all_nn=num_nn_variable(ind_Cd,ind_Se,ind_lig,all_dists,lig_cutoff,p_cutoff)

    cd_underc_ind=[]
    se_underc_ind=[]
    for i,n_nn in enumerate(all_nn):
        if ind_Cd[i]:
            cd_underc_ind.append(n_nn<nncutoff)
        elif ind_Se[i]:
            se_underc_ind.append(n_nn<nncutoff)

    cd_underc_ind=np.array(cd_underc_ind)
    se_underc_ind=np.array(se_underc_ind)

    return cd_underc_ind,se_underc_ind


def num_nn_variable(ind_Cd,ind_Se,ind_lig,dist_list,lig_cutoff,p_cutoff):
    '''
    Honestly not sure if Lexie already implemented this or not

    Does the coordination # counting, but uses different cutoffs for cation-ligand bonds and cation-anion bonds

    Inputs:
        ind_Cd: np array of which atoms are our cation, Natoms long
        ind_Se: np array of which atoms are our anion, Natoms long
        ind_lig: np array of which atoms are our ligand, Natoms long
        dist_list: Natoms x Natoms, the distance of each atom to each other atom. Includes self-distances
        lig_cutoff: the upper bond length cutoff for cation-ligand bonds, Angstroms
        p_cutoff: the upper bond length cutoff for anion-ligand bonds, Angstroms
    
    Outputs:
        nn_list: an array of the number of nearest neighbors for each atom. Size (Natoms,).
                 nn_list[i] = # of nearest neighbors for atom i
    '''
    nn_list=[]
    for i,atom in enumerate(dist_list):
        nn_count=0
        for j,dist in enumerate(atom):
            if i!=j: #this is how we're dealing with the self-bonds
                if ind_Cd[i]:
                    if ind_lig[j]:
                        if dist<lig_cutoff:
                            nn_count=nn_count+1
                    else:
                        if dist<p_cutoff:
                            nn_count=nn_count+1
                elif ind_Se[i]:
                    if dist<p_cutoff:
                            nn_count=nn_count+1
                elif ind_lig[i]:
                    if dist<lig_cutoff:
                            nn_count=nn_count+1
                else: #going to use the bigger cutoff for anything else
                    if dist<p_cutoff:
                            nn_count=nn_count+1
        nn_list.append(nn_count)

    nn_list=np.array(nn_list)


    return nn_list

def get_rms_distortion(new_coords,old_coords,target_index_new,target_index_old,cutoff):
    '''
    The goal of this function is to compute a sort of root-mean-squared deviation between the structure of an atom before and after a certain defect

    Centers the target atom in the same place and computes the difference between its bonded constituents

    Obviously won't work if the atom changes in coordination #
    Too lazy to code this to work with a change in the atom's indices

    Inputs:
        new_coords: np array of the coordinates of all atoms in the final structure, the one we are interested in
        old_coords: np array of the original (but still optimized) coordinates of all atoms before said defect
        target_index_new: integer, the overall index of the atom we are interested in in the new xyz
        target_index_old: integer, the overall index of the atom we are interested in in the old xyz
        cutoff: integer, a manual parameter that specifies what counts as "bonded"
    Outputs:
        rmsd: float, the rms deviation of the positions of all bonded atoms to our target
    '''

    new_dists=dist_all_points(new_coords) #all distances in new dot
    old_dists=dist_all_points(old_coords) #all distances in old dot

    #recenter coords
    new_cent_coords=new_coords-new_coords[target_index_new]
    old_cent_coords=old_coords-old_coords[target_index_old]

    target_new_dists=new_dists[target_index_new]
    target_old_dists=old_dists[target_index_old]

    bonded_indeces_new=[]
    for i,dist in enumerate(target_new_dists):
        if dist < cutoff and dist > 0:
            bonded_indeces_new.append(i)

    bonded_indeces_old=[]
    for i,dist in enumerate(target_old_dists):
        if dist < cutoff and dist > 0:
            bonded_indeces_old.append(i)


    #we now need to "match" the bonded atoms with themselves. We also get the distances for free
    deltas=[]
    for i,bond in enumerate(bonded_indeces_new):
        diffs=[]
        for j,old_bond in enumerate(bonded_indeces_old):
            diffs.append(np.sqrt(np.sum((new_cent_coords[bond] - old_cent_coords[old_bond])**2)))
        deltas.append((min(diffs)))
    deltas=np.array(deltas)

    #now compute the RMS
    num=np.sum(deltas**2)
    rms=np.sqrt(num/4)
    return rms



def connectivity_finder(dists,cutoff):
    '''
    The people have spoken and they say "more ezra code"

    makes an NAtoms list containing, for each atom, the indices of all other atoms within cutoff of that atom

    Inputs:
        dists: np array with distances between all atoms, size (Natoms, Natoms)
        cutoff: cutoff for a nearest neighbor distance
    Outputs:
        connect: NAtom alist of NBonded lists of python-adjusted indices for all other atoms within cutoff of that atom
    '''
    connectivity=[]
    for i,atom in enumerate(dists):
        bonds=[]
        for j,dist in enumerate(atom):
            
            if float(dist) < cutoff and i!=j:
                bonds.append(j)
        connectivity.append(bonds)

    return connectivity





def get_coplane(coords,dists,target_index,cutoff):
    '''
    Another iconic piece of ezra code

    finds a "coplanarity metric", i.e. some sort of distance from a plane (or line) for a given atom in a structure

    Inputs:
        coords: np array of xyz coordinates for the QD. Has shape (Natoms,3)
        dists: np array with distances between all atoms, size (Natoms, Natoms)
        target_index: the overall index of the atom in the .xyz file, adjusted to base 0 for python
        cutoff: cutoff for a nearest neighbor distance
    Outputs:
        the coplanarity metric. Defined differently for 2c, 3c, and 4c atoms
    '''
    bonded_indeces = []
    for i,atom in enumerate(coords):
        if dists[target_index][i] < cutoff and i != target_index: # and atom!="F": #for computing In/Ga-4c
            bonded_indeces.append(i)

    if len(bonded_indeces) > 3:
        #if len(bonded_indeces) == 5:
        #    print(target_index, "is 5 CMC")
        #print("Hope this is supposed to be 4-coordinate:")

        umbrella=1
        min_coplane = 10 #arbitrary high number
        for i,bond in enumerate(bonded_indeces): #loop through bonded atoms, find coplane metric excluding that atom
            v1 = coords[bonded_indeces[i-1]] - coords[bonded_indeces[i-2]]
            v2 = coords[bonded_indeces[i-3]] - coords[bonded_indeces[i-2]]
            norm_v = np.cross(v1,v2) #plane equation comes from normal vector to plane
            a = norm_v[0]
            b = norm_v[1]
            c = norm_v[2]
            d = -a*coords[bonded_indeces[i-1]][0]-b*coords[bonded_indeces[i-1]][1]-c*coords[bonded_indeces[i-1]][2]
            dist_from_plane = (a*coords[target_index][0]+b*coords[target_index][1]+c*coords[target_index][2]+d)/math.sqrt(a**2 + b**2 + c**2)

            if dist_from_plane<0: #sometimes it'll be negative
                dist_from_plane = dist_from_plane*-1

            #print(dist_from_plane, bond)

            if dist_from_plane < min_coplane:
                min_coplane=dist_from_plane

                i_dist=np.linalg.norm(coords[bonded_indeces[i]]-coords[target_index]) #dist of 4th point from center
                i_dist_from_plane=abs((a*coords[bonded_indeces[i]][0]+b*coords[bonded_indeces[i]][1]+c*coords[bonded_indeces[i]][2]+d)/math.sqrt(a**2 + b**2 + c**2))
                other_d1=np.linalg.norm(coords[bonded_indeces[i]]-coords[bonded_indeces[i-1]])
                other_d2=np.linalg.norm(coords[bonded_indeces[i]]-coords[bonded_indeces[i-2]])
                other_d3=np.linalg.norm(coords[bonded_indeces[i]]-coords[bonded_indeces[i-3]])
                if (i_dist > i_dist_from_plane) and (max(other_d1,other_d2,other_d3)<1.5*i_dist): #trying to distingish structures with umbrella like arrangements from the normal tetrahedron
                    umbrella=-1
                    #print(max(other_d1,other_d2,other_d3))

        return min_coplane*umbrella

    elif len(bonded_indeces) < 3:
        print("Hope this is supposed to be 2-coordinate:")

        v1 = coords[bonded_indeces[1]] - coords[bonded_indeces[0]]
        u1 = v1/np.linalg.norm(v1)
        d1=coords[target_index]-coords[bonded_indeces[0]]
        dist_vect=d1-np.dot(d1,u1)*u1
        dist_from_line=np.linalg.norm(dist_vect)

        return dist_from_line

    else:
        v1 = coords[bonded_indeces[1]] - coords[bonded_indeces[0]]
        v2 = coords[bonded_indeces[2]] - coords[bonded_indeces[0]]
        norm_v = np.cross(v1,v2) #plane equation comes from normal vector to plane
        a = norm_v[0]
        b = norm_v[1]
        c = norm_v[2]
        d = -a*coords[bonded_indeces[1]][0]-b*coords[bonded_indeces[1]][1]-c*coords[bonded_indeces[1]][2]
        dist_from_plane = (a*coords[target_index][0]+b*coords[target_index][1]+c*coords[target_index][2]+d)/math.sqrt(a**2 + b**2 + c**2)

        if dist_from_plane<0:
            dist_from_plane = dist_from_plane*-1

        return dist_from_plane


def get_off_tet_index(QD_xyz, ind_In, ind_P,ind_lig,in_underc_ind,p_underc_ind,dists,cutoff,angle_cutoff):
    '''
    This one's fully written by me

    Function that finds strongly off-tetrahedral 4 (and 5) Coordinate In and P atoms in a QD
    Defined as any 4-c P or In with at least three bond angles centered at that atom above angle_cutoff away from 109.5 degrees

    Inputs:
        xyz: np array of xyz coordinates for the QD. Has shape (Natoms,3)
        ind_In: boolean array of shape Natoms, indexing the In atoms
        ind_P: boolean array of shape Natoms, indexing the P atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        in_underc_ind: boolean array of shape NIndium, indexing the < 4-c In atoms
        p_underc_ind: boolean array of shape NPhosphorous, indexing the < 4-c P atoms 
        dists: np array with distances between all atoms, size (Natoms, Natoms)
        cutoff: cutoff for a nearest neighbor distance
        angle_cutoff: minimum degrees from 109.5 for an angle to be considered "off-tetrahedral"
    Outputs:
        in_off_tet_ind: boolean array of shape NIndium, indexing off-tetrahedral In-4c
        p_off_tet_ind: boolean array of shape NPhosphorous, indexing off-tetrahedral P-4c
        in_angles: list with angles centered on each In. Size (NIndium, Nangles). Unchanged from get_angles
        p_angles: list with angles centered on each P. Size (NPhosphorous, Nangles). Unchanged from get_angles
    '''

    in_angles,p_angles = get_angles(QD_xyz,ind_In, ind_P,dists,cutoff)

    in_off_tet_ind = []
    for i,indium in enumerate(in_angles):
        ok = True
        count = 0
        if in_underc_ind[i]:
            ok = False
            in_off_tet_ind.append(False)
        for j,angle in enumerate(indium):
            if abs(angle-109.5)>=angle_cutoff and ok and count < 1:
                count = count+1
            elif abs(angle-109.5)>=angle_cutoff and ok and count == 1:
                in_off_tet_ind.append(True)
                ok = False
            if j+1 == len(indium) and ok:
                in_off_tet_ind.append(False)                

    p_off_tet_ind = []
    for i,phos in enumerate(p_angles):
        ok = True
        count = 0
        if p_underc_ind[i]:
            ok = False
            p_off_tet_ind.append(False)
        for j,angle in enumerate(phos):
            if abs(angle-109.5)>=angle_cutoff and ok and count < 1:
                count = count+1
            elif abs(angle-109.5)>=angle_cutoff and ok and count == 1:
                p_off_tet_ind.append(True)
                ok = False
            if j+1 == len(phos) and ok:
                p_off_tet_ind.append(False) 

    #print(p_off_tet_ind)


    return in_off_tet_ind, p_off_tet_ind, in_angles, p_angles

def get_angles(QD_xyz,ind_In,ind_P,dists,cutoff):
    '''
    Another ezra classic

    Function that calculates all angles centered at each In and P atom between atoms within cutoff

    Inputs:
        QD_xyz: np array of xyz coordinates for the QD. Has shape (Natoms,3)
        ind_In: boolean array of shape Natoms, indexing the In atoms
        ind_P: boolean array of shape Natoms, indexing the P atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        dists: np array with distances between all atoms, size (Natoms, Natoms)
        cutoff: cutoff for a nearest neighbor distance
    Outputs: 
        in_angles: list with angles centered on each In. Size (NIndium, Nangles for that atom (depends on number of nearest neighbors))
        p_angles: list with angles centered on each P. Size (NPhosphorous, Nangles)
    '''

    #get all angles
    angles=[]
    for i,center in enumerate(QD_xyz):
        atom = []
        for j,atom1 in enumerate(QD_xyz):
            if dists[i][j]<cutoff and i!=j:
                for k,atom2 in enumerate(QD_xyz):
                    if dists[i][k]<cutoff and i!=k and j!=k and k>j:
                        dist1 = dists[i][j]
                        dist2 = dists[i][k]
                        dist3 = dists[j][k]
                        num = dist1**2 + dist2**2 - dist3**2
                        denom = 2*dist1*dist2
                        angle = math.acos(num/denom)
                        atom.append(math.degrees(angle))
        angles.append(atom)
    #all_angles = np.array(angles)

    in_angles = []
    p_angles = []
    for i,angle in enumerate(angles):
        if ind_In[i]:
            in_angles.append(angle)
        if ind_P[i]:
            p_angles.append(angle)


    return in_angles, p_angles

def get_off_pairs(dists,p_off_tet_ind,pp_cutoff,ind_P):
    '''
    written by ez

    Function that finds "adjacent" pairs of off-tetrahedral P (within pp_cutoff)

    Inputs:
        dists: np array with distances between all atoms, size (Natoms, Natoms)
        p_off_tet_ind: boolean array of shape NPhosphorous, indexing off-tetrahedral P-4c
        pp_cutoff: cutoff for two P to be considered adjacent, in Angstroms
        ind_P: boolean array of shape Natoms, indexing the P atoms
    Outputs:
        p_pairs:
    '''

    p_pairs = []
    p_count= -1
    #print(len(p_off_tet_ind))
    for i,atom in enumerate(dists):
        if ind_P[i]:
            #print("i",i)
            p_count=p_count+1
            if p_off_tet_ind[p_count]:
                p_count2 = -1
                for j,atom2 in enumerate(atom):
                    if ind_P[j]:
                        #print("j",j)
                        p_count2=p_count2+1
                        if p_off_tet_ind[p_count2] and atom2 < pp_cutoff and i != j and j > i:
                            p_pairs.append([p_count+1,p_count2+1])

    return p_pairs


def read_input_xyz(input_xyz):
    '''
    Function that reads xyz file into arrays

    Inputs: input_xyz -- .xyz file with the QD coordinates
    Outputs: xyz_coords -- np array with the coordinates of all the atoms (float)
             atom_names -- np array with the atom names (str)
    '''
    xyz_coords = np.loadtxt(input_xyz,skiprows=2,usecols=(1,2,3))
    atom_names = np.loadtxt(input_xyz,skiprows=2,usecols=(0,),dtype=str)
    return xyz_coords, atom_names

def write_xyz(out_file, atom_names, atom_xyz, comment=''):
    '''
    Function that writes xyz coordinate arrays to .xyz file

    Inputs: out_file   -- name of the file to write coordinates to
            atom_names -- np array or list of atom names (str)
            atom_xyz   -- np array or list of atom xyz coordinates (float)
            comment    -- comment line to write in the xyz file

    Outputs: Writes to xyz_file
    '''
    with open(out_file,'w') as xyz_file:
        xyz_file.write(str(len(atom_names))+'\n')
        xyz_file.write(comment+'\n')
        for i, atom in enumerate(atom_names):
            xyz_file.write('{:<2s}    {:< 14.8f}{:< 14.8f}{:< 14.8f}\n'.format(atom,atom_xyz[i][0],atom_xyz[i][1],atom_xyz[i][2]))
    return

def dist_all_points(xyz):
    '''
    Function that returns the distances between all atoms in an xyz file.

    Inputs:
        xyz: numpy array of xyz coordinates of atoms to calculate the
             distances between. Size (Natoms,3)

    Outputs:
        dist_array: array of distances between all atoms. Size (Natoms,Natoms).
                    dist_array[i][j] is the distance between atom i and atom j
    '''
    dists = [] # list to help build array
    for atom in xyz: # xyz = for each atom
        dist = np.sqrt(np.sum((atom - xyz)**2,axis=1)) # calc dist between atom(i) and all others
        dists.append(dist)
    dist_array = np.array(dists)
    return dist_array

def dist_atom12(all_dists,ind_1,ind_2):
    '''
    Function that returns an array of distances between two types of atoms.

    Inputs:
        all_dists: array of distances between all atoms, size (Natoms,Natoms)
        ind_1: array of boolean indices for first atom type (e.g. all Cd's)
        ind_2: array of boolean indices for second atom type (e.g. all Se's)

    Outputs:
        Returns a subset of all_dists that are the distances between atom
        type 1 and 2. Array of size (Natom1,Natom2)
    '''
    return all_dists[ind_1].T[ind_2].T

def get_dists_cs(QD_xyz,ind_Cd,ind_Se,ind_shell_cd,ind_shell_chal,ind_attach=False,ind_attach2=False):
    all_dists = dist_all_points(QD_xyz)
    cd_se_dists_all = dist_atom12(all_dists,ind_Cd,ind_Se) # cd (core) - se (core)
    se_cd_dists_all = dist_atom12(all_dists,ind_Se,ind_Cd) # se (core) - cd (core)
    # print(np.all(cd_se_dists_all==se_cd_dists_all.T))

    ind_ses =np.logical_or(ind_Se,ind_shell_chal) # index of se and s atoms
    ind_cdcd = np.logical_or(ind_Cd,ind_shell_cd) # index of all cd

    # print(ind_ses)

    cdcore_ses_dist = dist_atom12(all_dists,ind_Cd,ind_ses) # cd (core) - se and s
    secore_cd_dist  = dist_atom12(all_dists,ind_Se,ind_cdcd) # se (core) - cd (core) and cd (shell)
    cdshell_ses_dist = dist_atom12(all_dists,ind_shell_cd,ind_ses) # cd (shell) - se and s
    sshell_cd_dist = dist_atom12(all_dists, ind_shell_chal,ind_cdcd) # s (shell) - cd (core) and cd (shell)
    # print(all_dists,cd_se_dists_all,cdcore_ses_dist,secore_cd_dist,cdshell_ses_dist,sshell_cd_dist)

    # print(cdcore_ses_dist.shape)
    # print(sshell_cd_dist.shape)
    if np.any(ind_attach): # if ligands present
        ind_challig = np.logical_or(ind_ses,ind_attach)
        cd_lig_dists_all = dist_atom12(all_dists,ind_shell_cd,ind_attach)
        cd_chal_lig_dists_all = dist_atom12(all_dists,ind_shell_cd,ind_challig)
        # print(cd_lig_dists_all.shape)

    else: # may need to fix this so it works with non-lig
        cd_lig_dists_all = []
        cd_chal_lig_dists_all = cdshell_ses_dist # this should mean that anything that uses this will default to just cd-ses

    if np.any(ind_attach2):
        ind_cdlig = np.logical_or(ind_cdcd,ind_attach2)
        s_lig_dists_all = dist_atom12(all_dists,ind_shell_chal,ind_attach2)
        s_cd_lig_dists_all = dist_atom12(all_dists,ind_shell_chal,ind_cdlig)
    else:
        s_lig_dists_all = []
        s_cd_lig_dists_all = sshell_cd_dist # if no ligand 2, just return cd-s dists

    return all_dists,cd_se_dists_all,cdcore_ses_dist,secore_cd_dist,cdshell_ses_dist,sshell_cd_dist,cd_lig_dists_all,cd_chal_lig_dists_all,s_lig_dists_all,s_cd_lig_dists_all


def get_dists(QD_xyz,ind_Cd,ind_Se,ind_attach=False,ind_shell_cd=False,ind_shell_chal=False,cs=False):
    '''
    Function that calculates the distance between all atoms, as well as
    the distance between two types of atoms.

    Inputs:
        QD_xyz: xyz coordinates of all atoms in the QD (array size (Natoms,3))
        ind_Cd: indices of atom type 1 (e.g. Cd)
        ind_Se: indices of atom type 2 (e.g. Se)
        ind_attacH: (optional) indices of the ligand atoms that attach to Cd
                    (e.g. indices of N for MeNH2)

    Outputs:
        all_dists: np array with distances between all atoms, size (Natoms, Natoms)
        cd_se_dists_all: np array with distances between atom type 1 and atom
                         type 2 (e.g. Cd-Se distances only)
        se_cd_dists_all: np array with distances between atom type 2 and atom type 1
                         (e.g. Se-Cd distances) -- same as cd_se_dists_all but indexed
                         differently
        cd_lig_dists_all: only returned if ind_attach provided. distances between
                          cd atoms and ligand attach atoms
        cd_se_lig_dists_all: only returned if ind_attach provided. distances between
                             cd atoms and ligand attach atoms AND cd atoms and se atoms
    '''
    all_dists = dist_all_points(QD_xyz)



    cd_se_dists_all = dist_atom12(all_dists,ind_Cd,ind_Se)

    #print max distance for diameter purposes
    maxes = []
    for i,atom in enumerate(cd_se_dists_all):
        maxes.append(np.amax(atom))
    print("Diameter is", round(max(maxes),2))

    se_cd_dists_all = dist_atom12(all_dists,ind_Se,ind_Cd)

    if np.any(ind_attach): # if ligands present
        ind_selig = np.logical_or(ind_Se,ind_attach)
        cd_lig_dists_all = dist_atom12(all_dists,ind_Cd,ind_attach)
        cd_se_lig_dists_all = dist_atom12(all_dists,ind_Cd,ind_selig)

        return all_dists,cd_se_dists_all,cd_lig_dists_all,cd_se_lig_dists_all,se_cd_dists_all


    else:
        return all_dists,cd_se_dists_all,[],cd_se_dists_all,se_cd_dists_all

def get_dists_bonded(all_dists,ind_Cd,ind_Se):
    cdcd_dist=dist_atom12(all_dists,ind_Cd,ind_Cd)
    sese_dist=dist_atom12(all_dists,ind_Se,ind_Se)
    return cdcd_dist,sese_dist

def num_nn(dist_list,cutoff):
    '''
    Function that calculates the number of nearest neighbors that each atom has,
    based on a cutoff.

    Inputs:
        dist_list: array of distances between all atoms, size (Natoms,Natoms)
        cutoff: distance cutoff (in A) below which atoms are considered nearest
                neighbors/bonded

    Outputs:
        nn_list: an array of the number of nearest neighbors for each atom. Size (Natoms,).
                 nn_list[i] = # of nearest neighbors for atom i

    '''

    nn_list = np.sum(dist_list < cutoff,axis=1)
    return nn_list

def get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig=False,ind_lig2=False):
    '''
    Function that calculates the number of nearest neighbors for each atom,
    based on atom type. E.g. can restrict such that Cd only has Se NN's

    Inputs:
        cdselig_dists: distances between cd and se (or cd, and se + ligs)
        secd_dists: distances between se and cd
        ind_Cd: indices of cd atoms
        ind_Se: indices of se atoms
        cutoff: cutoff for NN interaction
        Natoms: number of atoms in the system

    Outputs:
        all_nn: an array of the number of nearest neighbors for each atom. Size (Natoms,).
                nn_list[i] = # of nearest neighbors for atom i
        cd_nn_selig: an array of the number of nearest neighbors for cd (size (Ncd,))
        se_nn_cdonly: an array of the number of nearest neighbors for se (size (Nse,))
    '''
    cd_nn_selig = num_nn(cdselig_dists,cutoff)
    se_nn_cdonly = num_nn(secd_dists,cutoff)

    all_nn = np.zeros(Natoms)
    all_nn[ind_Cd]=cd_nn_selig # using all nn for cd
    all_nn[ind_Se]=se_nn_cdonly # using just cd for se
    if np.any(ind_lig):
        all_nn[ind_lig]=100 # set these to very high to avoid any weirdness
    if np.any(ind_lig2):
        all_nn[ind_lig2]=100 # set these to very high to avoid any weirdness
    return all_nn,cd_nn_selig,se_nn_cdonly

def nn_histogram(xyz,ind_Cd,ind_Se,label1='',ind_attach=False,xyz2=False,label2=''):
    '''
    Function that makes a histogram of all nearest-neighbor distances in a QD (or comparing multiple!).

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array with the indices of Cd atoms in xyz
        ind_Se: boolean array with the indices of Se atoms in xyz
        label1: (optional) label for the legend of the histogram for xyz
        ind_attach: (optional) boolean array with the indices of attaching
                    ligand atoms in xyz (e.g. N for MeNH2)
        xyz2: (optional) np array of xyz coordinates for another QD to compare
        label2: (optional) label for the legend of the histogram for xyz2

    Outputs:
        plots a histogram of Cd-Se distances. If ind_attach is true, also
        plots histogram of Cd-ligand distances.
    '''

    all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyz,ind_Cd,ind_Se,ind_attach)

    if np.any(xyz2):
        all_dists2,cdse_dists2,cdlig_dists2,cdselig_dists2,secd_dists2 = get_dists(xyz2,ind_Cd,ind_Se,ind_attach)

    # Cd-Se distance histogram
    plt.figure()
    #plt.title("M-P distance")
    plt.title(label1+"-P distance")
    plt.hist(cdse_dists.flatten(),bins=800,label=label1) # crystal
    if np.any(xyz2): plt.hist(cdse_dists2.flatten(),bins=800,label=label2) # optimized
    if label2 !='': plt.legend()
    plt.xlim(1,4)
    # plt.show()
    #
    if np.any(ind_attach):
        # # Cd-ligand distance histogram
        plt.figure()
        plt.title(label1+"-ligand distance")
        plt.hist(cdlig_dists.flatten(),bins=800,label=label1)
        if np.any(xyz2): plt.hist(cdlig_dists2.flatten(),bins=800,label=label2)
        if label2 !='': plt.legend()
        plt.xlim(1,4)

    # plt.show()

    return


def parse_ind(atom_name,lig_attach="N"):
    '''
    Function to parse the indices for a quantum dot.

    Inputs:
        atom_name: np array with the atom names for the QD
        lig_attach: atom in the ligand that attaches to the Cd

    Returns:
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_CdSe: boolean array of shape Natoms, indexing both Cd and Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms (or, atoms that aren't Cd or Se)
        ind_selig: boolean array of shape Natoms, indexing the Se atoms and attach atoms
    '''
    ind_Cd = (atom_name == "Cd")
    ind_Se = (atom_name == "Se")
    ind_CdSe = np.logical_or(ind_Cd, ind_Se)
    ind_lig = np.logical_not(ind_CdSe)  # ligand atoms are defined as anything that isn't cd or se (!)
    ind_selig = np.logical_or(ind_Se,(atom_name == lig_attach))  # NOTE: not robust! only uses N, change for other ligands
    return ind_Cd, ind_Se, ind_CdSe, ind_lig, ind_selig

def get_underc_index(xyz,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False):
    '''
    Function that finds undercoordinated Cd and Se atoms in a QD.

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        ind_attach: boolean array indexing the ligand atoms that bind to Cd
        cutoff: cutoff for a nearest neighbor distance
        nncutoff: number of nearest neighbors to be considered "fully coordinated"
                  (< this classified as "undercoordinated")
        verbose: if True, prints the number of nearest neighbors for
                 "undercoordinated" atoms
    '''
    all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyz,ind_Cd,ind_Se,ind_attach)

    Natoms = len(ind_Cd)
    all_nn,cd_nn_se,se_nn_cd = get_nn(cdse_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)
    all_nn,cd_nn_selig,se_nn_cd = get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)


    #this section is currently configured for a specific structure
    cd_underc_ind = (cd_nn_selig)<nncutoff
    #for i,cad in enumerate(cd_underc_ind):
        #if i+1==4 or i+1==47 or i+1==50 or i+1==21 or i+1==15 or i+1==1 or i+1==38 or i+1==52 or i+1==20 or i+1==35 or i+1==32 or i+1==3:
    #    if i+1==40 or i+1==14 or i+1==27 or i+1==44 or i+1==2 or i+1==31 or i+1==51 or i+1==23 or i+1==9:
    #        cd_underc_ind[i] = True
    #    else:
    #        cd_underc_ind[i] = False

    se_underc_ind = se_nn_cd<nncutoff
    #for i,se in enumerate(se_underc_ind):
    #    #if i+1==3 or i+1==15 or i+1==17 or i+1==2 or i+1==5 or i+1==11 or i+1==24 or i+1==25 or i+1==28:
    #    if i+1==37 or i+1==38 or i+1==39 or i+1==40 or i+1==33 or i+1==23:
    #        se_underc_ind[i] = True
    #    else:
    #        se_underc_ind[i] = False

    if verbose:
        print('Undercoordinated Cd:',cd_nn_selig[cd_underc_ind])
        print('Undercoordinated Se:',se_nn_cd[se_underc_ind])
    return cd_underc_ind,se_underc_ind

def get_bonded_index(xyz,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False):
    '''
    Function that finds undercoordinated Cd and Se atoms in a QD.

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        ind_attach: boolean array indexing the ligand atoms that bind to Cd
        cutoff: cutoff for a nearest neighbor distance
        nncutoff: number of nearest neighbors to be considered "fully coordinated"
                  (< this classified as "undercoordinated")
        verbose: if True, prints the number of nearest neighbors for
                 "undercoordinated" atoms
    '''
    all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyz,ind_Cd,ind_Se,ind_attach)
    cdcd_dists,sese_dists=get_dists_bonded(all_dists,ind_Cd,ind_Se)
    Natoms = len(ind_Cd)
    all_nn,cd_nn_cd,se_nn_se = get_nn(cdcd_dists,sese_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)

    cd_bond_ind = cd_nn_cd>1
    se_bond_ind = se_nn_se>1

    if verbose:
        print('Bonded Cd:',cd_nn_cd[cd_bond_ind])
        print('Bonded Se:',se_nn_se[se_bond_ind])
    return cd_bond_ind,se_bond_ind



def get_ind_dif(xyz1,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,cutoff2=None,xyz2=None):
    '''
    Function to get indices of atoms that changed number of nearest neighbors.
    Can be in response to a different cutoff (in which case, supply cutoff2) or
    over an optimization (in which case, supply xyz2)

    Inputs:
        xyz1: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        ind_attach: boolean array indexing the ligand atoms that bind to Cd
        cutoff: cutoff for a nearest neighbor distance
        nncutoff: number of nearest neighbors to be considered "fully coordinated"
                  (< this classified as "undercoordinated")
        cutoff2: (optional) second cutoff to compare
        xyz2: (optional) second set of xyz coordinates to compare

    Outputs:
        ind_change_cd_pos: boolean array indexing the cd's that gain nearest neighbors
        ind_change_cd_neg: boolean array indexing the cd's that lose nearest neighbors
        ind_change_se_pos: boolean array indexing the se's that gain nearest neighbors
        ind_change_se_neg: boolean array indexing the se's that lose nearest neighbors
    '''
    all_dists1,cdse_dists1,cdlig_dists1,cdselig_dists1,secd_dists1 = get_dists(xyz1,ind_Cd,ind_Se,ind_attach)
    Natoms = len(ind_Cd)
    all_nn1,cd_nn_selig1,se_nn_cd1 = get_nn(cdselig_dists1,secd_dists1,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)
    if cutoff2:
        # distances all the same, just a different cutoff
        all_nn2,cd_nn_selig2,se_nn_cd2 = get_nn(cdselig_dists1,secd_dists1,ind_Cd,ind_Se,cutoff2,Natoms,ind_lig)
    elif np.any(xyz2):
        # different xyz, so different distances, but same cutoff
        all_dists2,cdse_dists2,cdlig_dists2,cdselig_dists2,secd_dists2 = get_dists(xyz2,ind_Cd,ind_Se,ind_attach)
        all_nn2,cd_nn_selig2,se_nn_cd2 = get_nn(cdselig_dists2,secd_dists2,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)

    cd_underc_ind1 = cd_nn_selig1<nncutoff
    se_underc_ind1 = se_nn_cd1<nncutoff
    cd_underc_ind2 = cd_nn_selig2<nncutoff
    se_underc_ind2 = se_nn_cd2<nncutoff

    nn_change_cd = cd_nn_selig2 - cd_nn_selig1
    nn_change_se = se_nn_cd2 - se_nn_cd1

    ind_change_cd_pos = nn_change_cd > 0
    ind_change_cd_neg = nn_change_cd < 0
    ind_change_se_pos = nn_change_se > 0
    ind_change_se_neg = nn_change_se < 0

    return ind_change_cd_pos,ind_change_cd_neg,ind_change_se_pos,ind_change_se_neg


def write_underc_xyz(xyz,atom_name,ind_Cd,ind_Se,cd_underc_ind,se_underc_ind,filestart,comment):
    '''
    Function to write the coordinates of undercoordinated atoms.

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        atom_name: array of atom names that correspond to xyz
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        cd_underc_ind: boolean array corresponding to atom_name, indexing
                       undercoordinated Cd's
        se_underc_ind: boolean array corresponding to atom_name, indexing
                       undercoordinated Se's
        filestart: most of the descriptive file name for the coordinates.
                       will have '_se.xyz' or '_cd.xyz' appended to it
        comment: comment for the xyz files
    Outputs:
        writes two xyz files: {filestart}_se.xyz with the coordinates of
        undercoordinated se's, and {filestart}_cd.xyz, with undercoordinated cd's
    '''
    cd_underc_name = atom_name[ind_Cd][cd_underc_ind]
    se_underc_name = atom_name[ind_Se][se_underc_ind]
    cd_underc_xyz = xyz[ind_Cd][cd_underc_ind]
    se_underc_xyz = xyz[ind_Se][se_underc_ind]

    write_xyz(filestart+'_se.xyz', se_underc_name, se_underc_xyz,comment)
    write_xyz(filestart+'_cd.xyz', cd_underc_name, cd_underc_xyz,comment)
    return



def get_underc_ind_large(ind_orig,ind_underc):
    '''
    Returns index for undercoordinated atom type, with the dimensions of the
    original number of atoms e.g. under coordinated Se index for Cd33Se33 will
    be len 33, this will return len 66 for use with other properties

    Inputs:
        ind_orig: index array for all atoms of X type (size: Natoms(total))
        ind_underc: index array for all undercoordinated atoms of X type
            (size: number of atoms of X type)

    Returns:
        large_underc_ind: index array for undercoordinated atoms of type X,
            mapped back to size of ind_orig (size: Natoms (total))
    '''
    large_underc_ind = copy.deepcopy(ind_orig)
    large_underc_ind[ind_orig] = ind_underc # USE INDICES FROM WHATEVER METHOD YOU PREFER
                                            # this is the undercoordinated at the end of the optimization
    return large_underc_ind

def sum_chargefrac(chargefrac_tot,ind_orig,ind_underc):
    '''
    Sums the charge fraction on the undercoordinated atoms and shapes array into (Nex,3)

    Inputs:
        chargefrac_tot: array of normalized charges on each atom
        ind_orig: index array for all atoms of X type (size: Natoms(total))
        ind_underc: index array for all undercoordinated atoms of X type
            (size: number of atoms of X type)

    Returns:
        sum_underc_reshape: array with summed charges on the undercoordinated
            atom for each excitation. Size (Nex,3) where col. 0 is electron,
            col. 1 is hole, col. 2 is delta (ignore)
    '''
    large_underc_ind = get_underc_ind_large(ind_orig,ind_underc)
    chargefrac_underc = chargefrac_tot[large_underc_ind]
    sum_chargefrac_underc= np.sum(chargefrac_underc,axis=0)
    # reshape so that we have an array of shape (Nex, 3) where column 0 is electron
    # charge sum, column 1 is hole charge sum, and column 2 is delta (ignored)
    sum_underc_reshape = np.reshape(sum_chargefrac_underc,(-1,3))
    return sum_underc_reshape


def print_indiv_ex(chargefrac_tot,ind_orig,ind_underc,n,atomname):
    '''
    Prints charge info about specific excitations and atom types

    Inputs:
        chargefrac_tot: array of normalized charges on each atom
        ind_orig: index array for all atoms of X type (size: Natoms(total))
        ind_underc: index array for all undercoordinated atoms of X type
            (size: number of atoms of X type)
        n: excitation number
        atomname: name of the atom (just for printing)
    '''
    large_underc_ind = get_underc_ind_large(ind_orig,ind_underc)
    chargefrac_underc = chargefrac_tot[large_underc_ind]
    sum_chargefrac_underc= np.sum(chargefrac_underc,axis=0)

    print('')
    print('Fraction of charge on each undercoordinated {} for excitation {}:'.format(atomname,n))
    print('   e           h')
    print(chargefrac_underc[:,3*n:3*n+2])
    print('')
    print('Sum of charge on undercoordinated {} for excitation {}:'.format(atomname,n))
    print('   e           h')
    print(sum_chargefrac_underc[3*n:3*n+2])

    max_ind = np.argmax(chargefrac_tot,axis=0) # index of the largest charge fraction on any atom
    max_charge=np.max(chargefrac_tot,axis=0)   # largest charge fraction on any atom
    print('')
    print('Largest charge fraction on any atom for excitation {}:'.format(n))
    print('   e           h')
    print(max_charge[3*n:3*n+2])
    print('')
    print('Is the largest charge fraction on an undercoordinated {}?'.format(atomname))
    print('   e     h')
    print(np.any(chargefrac_underc[:,3*n:3*n+2]==max_charge[3*n:3*n+2],axis=0))
    # print(atom_name_start[max_ind][3*n:3*n+3]) # atom name with largest charge fraction

    # creates an array (Nex, 3) where each entry is whether the max charge fraction is on an undercoordinated se
    # found this wasn't useful because it's almost always on it, even for bulk excitations
    max_is_underc_long = np.any(chargefrac_underc==max_charge,axis=0)
    max_is_underc= np.reshape(max_is_underc_long,(-1,3))
    # print(max_is_underc[100:120])

    # finds the top 5 highest charge fractions on any atom
    top5_ind = np.argpartition(-chargefrac_tot,5,axis=0)[:5] # index of top 5
    top5 = np.take_along_axis(chargefrac_tot,top5_ind,axis=0) # value of top 5
    print('')
    print('Top 5 largest charge fractions on any atom for excitation {}:'.format(n))
    print('   e           h')
    print(top5[:,3*n:3*n+2])

    return

#######
#
# CORE-SHELL SPECIFIC FUNCTIONS
#
#######
def cs(ind_Cd_core,ind_Se,ind_Cd_shell,ind_S,cutoff,nncutoff,dist_list,ind_attach=False,ind_attach2=False,verbose=False):
    '''
    Function that finds undercoordinated Cd and Se atoms in a QD.

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        ind_attach: boolean array indexing the ligand atoms that bind to Cd
        cutoff: cutoff for a nearest neighbor distance
        nncutoff: number of nearest neighbors to be considered "fully coordinated"
                  (< this classified as "undercoordinated")
        verbose: if True, prints the number of nearest neighbors for
                 "undercoordinated" atoms
    '''
    all_dists,cdse_core_dists,cdcore_ses_dist,secore_cd_dist,cdshell_ses_dist,sshell_cd_dist,cdshell_lig_dist,cdshell_chal_lig_dist,sshell_lig_dist,sshell_cd_lig_dist = dist_list #get_dists_cs(xyz,ind_Cd_core,ind_Se,ind_Cd_shell,ind_S)
    # print(cdse_core_dists)
    # print(cdse_core_dists.T)
    Natoms = len(ind_Cd_core)
    # print(Natoms)
    all_nn,cd_se_nn,se_cd_nn = get_nn(cdse_core_dists,cdse_core_dists.T,ind_Cd_core,ind_Se,cutoff,Natoms) # bare core
    all_nn,cdcore_nn_ses,se_nn_cdcd = get_nn(cdcore_ses_dist,secore_cd_dist,ind_Cd_core,ind_Se,cutoff,Natoms)    # core coord. when considering shell too
    all_nn,cdshell_nn_ses,s_nn_cdcd = get_nn(cdshell_chal_lig_dist,sshell_cd_lig_dist,ind_Cd_shell,ind_S,cutoff,Natoms,ind_attach,ind_attach2) # undercoordinated shell atoms (includes core-shell bonds)

    # core-core
    cd_underc_ind = cd_se_nn < nncutoff
    se_underc_ind = se_cd_nn < nncutoff

    # core-core&shell
    cd_underc_inclshell_ind = cdcore_nn_ses < nncutoff
    se_underc_inclshell_ind = se_nn_cdcd < nncutoff

    # shell - core&shell
    cdshell_underc_inclcore_ind = cdshell_nn_ses < nncutoff
    s_underc_inclcore_ind = s_nn_cdcd < nncutoff

    if len(cdshell_lig_dist) > 0:
        all_nn,cdshell_nn_lig,lig_nn_cdshell = get_nn(cdshell_lig_dist,cdshell_lig_dist.T,ind_Cd_shell,ind_attach,cutoff,Natoms)

        # ligand-Cd
        attach_underc_ind = lig_nn_cdshell < 1 # each N should be bound to one cd
    else:
        attach_underc_ind = []
    if len(sshell_lig_dist) > 0:
        all_nn,sshell_nn_lig,lig_nn_sshell = get_nn(sshell_lig_dist,sshell_lig_dist.T,ind_S,ind_attach2,cutoff,Natoms)

        # ligand-Cd
        attach_underc_ind2 = lig_nn_sshell < 1 # each N should be bound to one cd
    else:
        attach_underc_ind2 = []
    if verbose:
        print('Undercoordinated Cd (core only):',cd_se_nn[cd_underc_ind])
        print('Undercoordinated Se (core only):',se_cd_nn[se_underc_ind])
        print('Undercoordinated Cd (core with shell):',cdcore_nn_ses[cd_underc_inclshell_ind])
        print('Undercoordinated Se (core with shell):',se_nn_cdcd[se_underc_inclshell_ind])
        print('Undercoordinated Cd (shell with core):',cdshell_nn_ses[cdshell_underc_inclcore_ind])
        print('Undercoordinated Se (shell with core):',s_nn_cdcd[s_underc_inclcore_ind])
    return cd_underc_ind,se_underc_ind,cd_underc_inclshell_ind,se_underc_inclshell_ind,cdshell_underc_inclcore_ind,s_underc_inclcore_ind,attach_underc_ind,attach_underc_ind2

# def get_bonded_index_cs(xyz,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False):
def get_bonded_index_cs(all_dists,ind_Cd,ind_Se,ind_cd_shell,ind_s_shell,cutoff,verbose=False):
    '''
    Function that finds undercoordinated Cd and Se atoms in a QD.

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        ind_attach: boolean array indexing the ligand atoms that bind to Cd
        cutoff: cutoff for a nearest neighbor distance
        nncutoff: number of nearest neighbors to be considered "fully coordinated"
                  (< this classified as "undercoordinated")
        verbose: if True, prints the number of nearest neighbors for
                 "undercoordinated" atoms
    '''
    # all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyz,ind_Cd,ind_Se,ind_attach)
    ind_cd_all = np.logical_or(ind_Cd,ind_cd_shell)
    ind_chal = np.logical_or(ind_Se, ind_s_shell)
    cdcd_core_dists,sese_core_dists=get_dists_bonded(all_dists,ind_Cd,ind_Se)
    cdcd_cs_dists,ses_cs_dists=get_dists_bonded(all_dists,ind_cd_all,ind_chal)
    cdcd_shell_dists,ss_shell_dists=get_dists_bonded(all_dists,ind_cd_shell,ind_s_shell)
    Natoms = len(ind_Cd)

    all_nn,cd_nn_cd_core,se_nn_se_core = get_nn(cdcd_core_dists,sese_core_dists,ind_Cd,ind_Se,cutoff,Natoms)
    all_nn,cd_nn_cd_cs,se_nn_s_cs = get_nn(cdcd_cs_dists,ses_cs_dists,ind_cd_all,ind_chal,cutoff,Natoms)
    all_nn,cd_nn_cd_shell,s_nn_s_shell = get_nn(cdcd_shell_dists,ss_shell_dists,ind_cd_shell,ind_s_shell,cutoff,Natoms)

    cd_core_bond_ind = cd_nn_cd_core>1
    se_core_bond_ind = se_nn_se_core>1

    cd_cs_bond_ind = cd_nn_cd_cs>1
    ses_cs_bond_ind = se_nn_s_cs>1

    cd_shell_bond_ind = cd_nn_cd_shell>1
    s_shell_bond_ind = s_nn_s_shell>1


    if verbose:
        print('Bonded Cd:',cd_nn_cd[cd_bond_ind])
        print('Bonded Se:',se_nn_se[se_bond_ind])
    return cd_core_bond_ind,se_core_bond_ind,cd_cs_bond_ind,ses_cs_bond_ind,cd_shell_bond_ind,s_shell_bond_ind

