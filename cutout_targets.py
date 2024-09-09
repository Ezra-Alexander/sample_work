import numpy as np
import sys
import numpy.linalg as npl
import copy
from geom_helper import *
import os
from qchem_helper import get_all_alpha, get_ao_ind, get_ind_ao_underc, get_alpha

def main():
	#generates .xyz cutouts of the center of a list of specified molecular orbitals
	#this doesn't let you specify which atom in the QD to cutout, it just takes the 4c atom with the highest alpha in the state

	xyz_file=sys.argv[1] #your .xyz file
	bas_file = sys.argv[2]      # txt file with total number of orbitals and occupied orbitals
	coeff_file = sys.argv[3]    # txt version of qchem 53.0 OR numpy version
	mo_indices=[int(x)-1 for x in sys.argv[4:]] #the indices of the MOs to get cutouts for

	dir_name="test" #the name of the directory the .xyz cutouts will be written to

	orb_per_atom_def2svp={'In': 26, 'Ga': 32, 'P': 18, 'Cl': 18, 'Br': 32,'F':14,'H':5,'O':14, 'C':14,'S':18, 'Li':9, 'Al':18, "Zn":31,"S":18,"Se":32, "Si":18}
	orb_per_atom=orb_per_atom_def2svp # choose which dictionary to use

	# parse orbital info
	nbas,nocc=np.loadtxt(bas_file,dtype=int,unpack=True)
	homo=nocc-1

	# read xyz file
	coords,atoms=read_input_xyz(xyz_file)

	# read 53.npz
	coeff_file_expand=np.load(coeff_file)
	mo_mat=coeff_file_expand['arr_0'] # normalized
	mo_e = coeff_file_expand['arr_1']
	mo_e = mo_e * 27.2114 # MO energy, in eV

	#step 1 is to identify the under-coordinated atoms in our QD
	#the end result needs to be those boolean numpy arrays to feed into get_ind_ao_underc to feed into get_alpha

	ind_In = (atoms=='In')
	ind_P = np.logical_or((atoms=='P'), np.logical_or((atoms=='S'),(atoms=='Se')))
	ind_Ga = (atoms=='Ga')
	ind_Al=np.logical_or((atoms=="Al"),(atoms=="Zn"))
	ind_cat = np.logical_or(ind_In, np.logical_or(ind_Ga,ind_Al))
	ind_lig = np.logical_or(atoms=="F",np.logical_or((atoms=="Cl"),(atoms=="H")))
	ind_other = np.logical_not(np.logical_or(ind_cat,np.logical_or(ind_P,ind_lig)))
	if np.count_nonzero(ind_other)>0:
		raise Exception("There's an element here that isn't supported yet!")

	dists=dist_all_points(coords)
	connectivity=smart_connectivity_finder(dists,atoms)
	angles=get_close_angles(coords,dists,connectivity)
	bond_avs=average_bond_lengths(atoms,dists,connectivity)

	ind_cat_uc, ind_p_uc = get_ind_uc(atoms,connectivity,(ind_cat,ind_P))

	print("Finished processing geometry")

	ind_cat_ao,ind_p_ao,ind_lig_ao = get_ao_ind([ind_cat,ind_P,ind_lig],atoms,nbas,orb_per_atom)
	cat_underc_ind_ao,p_underc_ind_ao = get_ind_ao_underc(atoms,nbas,orb_per_atom,ind_cat_uc,ind_p_uc)

	p_nouc_ao = np.logical_xor(p_underc_ind_ao,ind_p_ao)
	cat_nouc_ao = np.logical_xor(cat_underc_ind_ao,ind_cat_ao)

	alpha_cat_uc,alpha_cat_fc,alpha_p_uc,alpha_p_fc,alpha_lig=get_alpha(mo_mat,[cat_underc_ind_ao,cat_nouc_ao,p_underc_ind_ao,p_nouc_ao,ind_lig_ao])

	test = np.all(np.isclose(alpha_cat_uc+alpha_cat_fc+alpha_p_uc+alpha_p_fc+alpha_lig,1))
	if test == False: raise ValueError('Alpha doesnt add to 1!')
	print('Alphas add to 1?:',test)

	atoms_by_orbs=get_all_alpha(atoms,nbas,mo_mat,orb_per_atom)
	orbs_by_atoms=atoms_by_orbs.T

	n_p=np.count_nonzero(ind_P)
	n_cat=np.count_nonzero(ind_cat)
	alpha_p=alpha_p_fc+alpha_p_uc
	alpha_cat=alpha_cat_fc+alpha_cat_uc
	n_p_uc=np.count_nonzero(ind_p_uc)
	n_cat_uc=np.count_nonzero(ind_cat_uc)

	if not os.path.exists(dir_name):
  		os.mkdir(dir_name)

	for mo in mo_indices:
		make_cutout(mo,atoms,coords,orbs_by_atoms[mo],connectivity,dir_name)


def make_cutout(mo,atoms,coords,orb_by_atoms,connectivity,dir_name):
	'''
	Makes a .xyz file of the predominant 4c atom in a specified MO
	Inputs:
		mo: int. the index of the mo to target, 0-based
		atoms: np array of strs of element symbols. len Natoms
		coords: np array of cartesian coordinates. len NAtoms x 3
		orb_by_atoms: np array of alpha for each atom in the target mo. len NAtoms
		connectivity: np array of the connectivity graph. len NAtoms x CoordN
		dir_name: str. the path to the directory to write the cutout in
	'''

	#determine the max alpha 4c atom

	coordination_numbers = np.array([len(connectivity[i]) for i in range(len(atoms))])
	mask=coordination_numbers==4
	max_4c=np.max(orb_by_atoms[mask])
	max_ind=np.where(orb_by_atoms==max_4c)[0][0]
	
	coords_2_write=[coords[max_ind]]
	atoms_2_write=[atoms[max_ind]]
	for i in connectivity[max_ind]:
		coords_2_write.append(coords[i])
		atoms_2_write.append(atoms[i])

	name=str(mo+1)+"_cutout_1.xyz"
	print("Writing",name)
	write_xyz(dir_name+"/"+name,atoms_2_write,coords_2_write)


if __name__ == "__main__":
	main()