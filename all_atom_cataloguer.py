import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy.linalg as npl
import copy
from geom_helper import *
from pdos_helper import dos_grid_general,get_alpha,get_ao_ind, get_ind_ao_underc
import pandas as pd
from qchem_helper import get_dipole, get_total_lowdin, get_all_alpha
import os
import time


def main():
	#This script is a descendant of the structural_trap_finder.py
	#It aims to process and catalog the DFT results for a QD
	#but instead of cataloguing structural traps, it catalogs by atom
	#It does not generate cutout .xyzs

	#each non-ligand atom is associated with the state closest to the band gap in which it has near maximal contribution (controlled by a parameter)
	#it assigns each of these states one of three labels: bulk, uc trap, or structural trap
		#It can read in band edges or find them on its own
		#it assumes P, S, and Se give rise to occupied states and In, Ga, Al, Zn give rise to unoccupied states
		#it ignores the QC state
		#note that this implementation allows multiple atoms to map to one state. This is an inevitable consequence of how mixed these states are

	#The catalog then includes with each atom info about its N closest neighbors (controlled by a parameter)

	#Data is written to a local excel file which can then be exported to a centralized excel file
	file_name="all_atom_catalogue.xlsx"

	xyz_file=sys.argv[1] #your .xyz file
	bas_file = sys.argv[2]      # txt file with total number of orbitals and occupied orbitals
	coeff_file = sys.argv[3]    # txt version of qchem 53.0 OR numpy version
	ipr_file = sys.argv[4]	# csv file with the ipr (now formally Paticipation Ratio) for each MO
	out_file=sys.argv[5] #for mulliken populations, mainly. the ipr.out or plot.out
	low_orb=sys.argv[6] #low_orb.npy, from ipr analysis. For Lowdin populations.
	if len(sys.argv)>7: #if you don't enter manual band edges, it will form an estimate from the PDOS and PR
		occ_min=int(sys.argv[7]) #the relative index of the 1st occupied bulk state. If HOMO-17, for example, put 17
		vir_max=int(sys.argv[8]) #the relative index of the 1st virtual bulk state. If LUMO+17, for example, put 17
		find_band_edges=False
	else:
		find_band_edges=True

	alpha_thresh=0.9 #an arbitrary parameter (0-1). what fraction of the max alpha in a state for an atom to have for that state to also belong to that atom
	ipr_thresh=0.5 #an arbitrary parameter (0-1). the fraction of the "max" pr that a state could have and still be considered a bulk state
	neighbors=52 #the number of nearest neighbors to include in the saved coordinate info. better to be big
	angle_cutoff=16 #saves all angles between closest angle_cutoff neighbors

	covalent_radii={"In":1.42,"Ga":1.22,"P":1.07,"Cl":1.02,"F":0.57,"Zn":1.22,"S":1.05,"Se":1.20,"O":0.66,"Al":1.21,"H":0.31}
	atomic_numbers={'In':21,"Ga":31,"P":15,"Cl":17,"Br":35,"F":9,"H":1,"O":8,"C":6,"S":16,"Li":3,"Al":13,"Zn":30,"Se":34,"Si":14} #wih ecps included
	orb_per_atom_def2svp={'In': 26, 'Ga': 32, 'P': 18, 'Cl': 18, 'Br': 32,'F':14,'H':5,'O':14, 'C':14,'S':18, 'Li':9, 'Al':18, "Zn":31,"S":18,"Se":32, "Si":18}
	orb_per_atom=orb_per_atom_def2svp # choose which dictionary to use

	# parse orbital info
	nbas,nocc=np.loadtxt(bas_file,dtype=int,unpack=True)
	homo=nocc-1

	# read xyz file
	coords,atoms=read_input_xyz(xyz_file)
	coords=center_coords(coords)

	# read 53.npz
	coeff_file_expand=np.load(coeff_file)
	mo_mat=coeff_file_expand['arr_0'] # normalized
	mo_e = coeff_file_expand['arr_1']
	mo_e = mo_e * 27.2114 # MO energy, in eV

	#read low_orb_ipr.csv
	ipr=np.loadtxt(ipr_file)

	#read mulliken pops and dipole from ipr.out
	mulliken=get_mulliken(out_file,atoms)
	mulliken_averages=average_over_element(atoms,mulliken)
	dipole=get_dipole(out_file)

	# read in lowdin charges per orbital. includes virtuals
	low_per_orb=np.load(low_orb)
	total_lowdins=get_total_lowdin(low_per_orb,atoms,nocc)
	lowdin_averages=average_over_element(atoms,total_lowdins)

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

	# I also need to hold on to information about what particular atoms contribute to each MO
	# this is such a stupid way of doing this
	# every_atom=[]
	# for i,atom in enumerate(atoms):
	# 	ind_i=np.full(len(atoms),False)
	# 	ind_i[i]=True
	# 	every_atom.append(ind_i)
	# every_atom=np.array(every_atom)

	# every_atom_ao=get_ao_ind(every_atom,atoms,nbas,orb_per_atom)
	# alpha_total=get_alpha(mo_mat,every_atom_ao)
	atoms_by_orbs=get_all_alpha(atoms,nbas,mo_mat,orb_per_atom)
	orbs_by_atoms=atoms_by_orbs.T

	n_p=np.count_nonzero(ind_P)
	n_cat=np.count_nonzero(ind_cat)
	alpha_p=alpha_p_fc+alpha_p_uc
	alpha_cat=alpha_cat_fc+alpha_cat_uc
	n_p_uc=np.count_nonzero(ind_p_uc)
	n_cat_uc=np.count_nonzero(ind_cat_uc)

	#if I didn't set band edges, I need to estimate them here
	if find_band_edges:
		occ_min,vir_max,vbm_e,cbm_e,qc_state,qc_e=band_edge_finder(ipr,alpha_p,atoms,alpha_p_fc,mo_e,alpha_cat,alpha_cat_fc,homo,ipr_thresh,n_p_uc,n_cat_uc)

		print("VBM: HOMO-"+str(occ_min)+" ("+str(homo-occ_min+1)+"), energy "+str(round(vbm_e,2)))
		print("CBM: LUMO+"+str(vir_max)+" ("+str(nocc+vir_max+1)+"), energy "+str(round(cbm_e,2)))
		if qc_state:
			print("Quantum-confined state at",str(round(qc_e,2)))
		else:
			print("NO quantum-confined state")

	#up until this point, everything is more or less the same as the structural trap finder
	#now we want to do the loop over each atom
	chosen_states,closest_atoms,state_labels,chosen_atoms,tetrahedral_errors,seesaw_errors,n_lig=catalogue_all_atoms(atoms,ind_lig,ind_cat,ind_P,alpha_cat,ipr,orbs_by_atoms,dists,angles,connectivity,nocc,ipr_thresh,alpha_thresh,neighbors,vir_max,occ_min)

	#now let's write everything to a local excel file
	#let's use pandas this time
	all_atoms_to_excel(file_name,neighbors,angle_cutoff,chosen_states,homo,state_labels,vir_max,mo_e,occ_min,ipr,alpha_cat_uc,alpha_cat_fc,alpha_p_uc,alpha_p_fc,alpha_lig,alpha_thresh,ipr_thresh,n_cat,n_p,ind_lig,atoms,ind_In,qc_state,dipole,connectivity,chosen_atoms,orbs_by_atoms,mulliken,mulliken_averages,low_per_orb,total_lowdins,lowdin_averages,coords,tetrahedral_errors,seesaw_errors,n_lig,closest_atoms,angles,dists,bond_avs)

def all_atoms_to_excel(file_name,neighbors,angle_cutoff,chosen_states,homo,state_labels,vir_max,mo_e,occ_min,ipr,alpha_cat_uc,alpha_cat_fc,alpha_p_uc,alpha_p_fc,alpha_lig,alpha_thresh,ipr_thresh,n_cat,n_p,ind_lig,atoms,ind_In,qc_state,dipole,connectivity,chosen_atoms,orbs_by_atoms,mulliken,mulliken_averages,low_per_orb,total_lowdins,lowdin_averages,coords,tetrahedral_errors,seesaw_errors,n_lig,closest_atoms,angles,dists,bond_avs):
	'''
	Writes all the listed data about all of the chosen atoms to an excel file
	'''
	nocc=homo+1
	#calculate vbm_e and cbm_e
	vbm_e=mo_e[homo-occ_min]
	cbm_e=mo_e[nocc+vir_max]

	labels=[]

	#State info
	labels.append("MO Index") 
	labels.append("Occupied?")
	labels.append("State Label")
	labels.append("Trapping?")
	labels.append("#States from VBM/CBM Edge (Positive for Trap)")
	labels.append("Energy from VBM/CBM Edge (Positive for Trap)")
	labels.append("Participation Ratio")
	labels.append("State Energy (eV)")
	labels.append("Total Under-Coordinated Cation Alpha")
	labels.append("Total Fully-Coordinated Cation Alpha")
	labels.append("Total Under-Coordinated Anion Alpha")
	labels.append("Total Fully-Coordinated Anion Alpha")
	labels.append("Total Ligand Alpha")
	#QD info
	labels.append("Employed Max Alpha Contribution Threshold")
	labels.append("Employed State-Specific IPR Threshold")
	labels.append("Total # Cations")
	labels.append("Total # Anions")
	labels.append("Total # Ligand")
	labels.append("Total # Atoms")
	labels.append("Fraction of Cations that are In")
	labels.append("Total Charge")
	labels.append("HOMO-LUMO Gap (eV)")
	labels.append("VBM relative index")
	labels.append("CBM relative index")
	labels.append("Bulk-Bulk Band Gap (eV)")
	labels.append("Quantum Confined State?")
	labels.append("Total Dipole Moment")
	labels.append("Total Structural Traps in QD")
	labels.append("Total Under-Coordinated Traps in QD")
	#Center info
	labels.append("Central Element")
	labels.append("Center Element-Index")
	labels.append("Center Total-Index")
	labels.append("Center Coordination #")
	labels.append("Center Alpha")
	labels.append("Center Mulliken")
	labels.append("Center Mulliken Relative to Element Average")
	labels.append("Center Lowdin in State")
	labels.append("Center Total Lowdin")
	labels.append("Center Total Lowdin Relative to Element Average")
	labels.append("Center Dipole Overlap")
	#Bound element info
	labels.append("Tetrahedral Absolute Error")
	labels.append("Tetrahedral Squared Error")
	labels.append("SeeSaw Absolute Error")
	labels.append("SeeSaw Squared Error")
	labels.append("Number of Bound Ligand")
	labels.append('Bond Angle 1') # 6 closest bond angles, largest to smallest
	labels.append('Bond Angle 2') # 6 closest bond angles, largest to smallest
	labels.append('Bond Angle 3') # 6 closest bond angles, largest to smallest
	labels.append('Bond Angle 4') # 6 closest bond angles, largest to smallest
	labels.append('Bond Angle 5') # 6 closest bond angles, largest to smallest
	labels.append('Bond Angle 6') # 6 closest bond angles, largest to smallest
	labels.append('Relative Bond Length 1')# 4 closest adjusted bond lengths, smallest to largest. Relative to QD average
	labels.append('Relative Bond Length 2')# 4 closest adjusted bond lengths, smallest to largest. Relative to QD average
	labels.append('Relative Bond Length 3')# 4 closest adjusted bond lengths, smallest to largest. Relative to QD average
	labels.append('Relative Bond Length 4')# 4 closest adjusted bond lengths, smallest to largest. Relative to QD average

	#we then loop over each nearby atom, closest to farthest, recording
	for i in range(neighbors):
		labels.append('Neighbor '+str(i)+" Element")
		labels.append('Neighbor '+str(i)+" Element-Index")
		labels.append('Neighbor '+str(i)+" Total Index")
		labels.append('Neighbor '+str(i)+" Coordination Number")
		labels.append('Neighbor '+str(i)+" Distance to Center")
		labels.append('Neighbor '+str(i)+" Alpha")
		labels.append('Neighbor '+str(i)+" Mulliken Charge")
		labels.append('Neighbor '+str(i)+" Mulliken Relative to Element Average")
		labels.append('Neighbor '+str(i)+" Lowdin in State")
		labels.append('Neighbor '+str(i)+" Total Lowdin Charge")
		labels.append('Neighbor '+str(i)+" Total Lowdin Charge Relative to Element Average")
		labels.append('Neighbor '+str(i)+" Projected X Coordinate") #basis is shortest bond, coplanar codirectional orthogonal to 2nd shortest bond, and their cross product
		labels.append('Neighbor '+str(i)+" Projected Y Coordinate")
		labels.append('Neighbor '+str(i)+" Projected Z Coordinate")
		if i>0:
			labels.append('Neighbor '+str(i)+" Angle to Center to Center Bond 1")
		if i>1:
			labels.append('Neighbor '+str(i)+" Dihedral Angle to Center Bond 2 to Center to Center Bond 1")
		for j in range(neighbors):
			if i>j and j>0 and i<angle_cutoff:
				labels.append('Neighbor '+str(i)+" Neighbor "+str(j)+" Angle with Center")

	data=[]
	for i,state in enumerate(chosen_states):
		center_data=[]

		center_data.append(state+1) #labels.append("MO Index")

		if state>homo: #labels.append("Occupied?")
			center_data.append(0)
		else:
			center_data.append(1)

		center_data.append(state_labels[i]) #labels.append("State Label")

		if state_labels[i]!="Bulk":	#labels.append("Trapping?")
			center_data.append(1)
		else:
			center_data.append(0)

		if state>homo: 
			center_data.append(vir_max-(state-nocc)) #labels.append("#States from VBM/CBM Edge (Positive for Trap)")
			center_data.append(cbm_e-mo_e[state]) #labels.append("Energy from VBM/CBM Edge (Positive for Trap)")
		else:
			center_data.append(occ_min-(homo-state))
			center_data.append(mo_e[state]-vbm_e)

		center_data.append(ipr[state]) #labels.append("Participation Ratio")

		center_data.append(mo_e[state]) #labels.append("State Energy (eV)")

		center_data.append(alpha_cat_uc[state]) #labels.append("Total Under-Coordinated Cation Alpha")

		center_data.append(alpha_cat_fc[state]) #labels.append("Total Fully-Coordinated Cation Alpha")

		center_data.append(alpha_p_uc[state]) #labels.append("Total Under-Coordinated Anion Alpha")

		center_data.append(alpha_p_fc[state]) #labels.append("Total Fully-Coordinated Anion Alpha")

		center_data.append(alpha_lig[state]) #labels.append("Total Ligand Alpha")

		center_data.append(alpha_thresh) #labels.append("Employed Max Alpha Contribution Threshold")

		center_data.append(ipr_thresh) #labels.append("Employed State-Specific IPR Threshold")

		center_data.append(n_cat) #labels.append("Total # Cations")

		center_data.append(n_p) #labels.append("Total # Anions")

		center_data.append(np.count_nonzero(ind_lig)) #labels.append("Total # Ligand")

		center_data.append(len(atoms)) #labels.append("Total # Atoms")

		center_data.append(np.count_nonzero(ind_In)/n_cat) #labels.append("Fraction of Cations that are In")

		center_data.append((3*n_cat)-(3*n_p)-np.count_nonzero(ind_lig)) #labels.append("Total Charge")

		center_data.append(mo_e[homo+1]-mo_e[homo]) #labels.append("HOMO-LUMO Gap (eV)")

		center_data.append(occ_min) #labels.append("VBM relative index")

		center_data.append(vir_max) #labels.append("CBM relative index")

		center_data.append(mo_e[homo+1+vir_max]-mo_e[homo-occ_min]) #labels.append("Bulk-Bulk Band Gap (eV)")

		if qc_state:
			center_data.append(1) #labels.append("Quantum Confined State?")
		else:
			center_data.append(0)

		center_data.append(np.linalg.norm(dipole)) #labels.append("Total Dipole Moment")

		center_data.append(state_labels.count("Structural")) #labels.append("Total Structural Traps in QD")

		center_data.append(state_labels.count("Under-coordinated")) #labels.append("Total Under-Coordinated Traps in QD")

		center_data.append(atoms[chosen_atoms[i]]) #labels.append("Central Element")

		center_data.append(to_atom_specific_index(atoms,chosen_atoms[i])) #labels.append("Center Element-Index")

		center_data.append(chosen_atoms[i]+1) #labels.append("Center Total-Index")

		center_data.append(len(connectivity[chosen_atoms[i]])) #labels.append("Center Coordination #")

		center_data.append(orbs_by_atoms[state][chosen_atoms[i]]) #labels.append("Center Alpha")

		center_data.append(mulliken[chosen_atoms[i]]) #labels.append("Center Mulliken")

		center_data.append(mulliken[chosen_atoms[i]]-mulliken_averages[atoms[chosen_atoms[i]]]) #labels.append("Center Mulliken Relative to Element Average")

		center_data.append(low_per_orb[chosen_atoms[i]][state]) #labels.append("Center Lowdin in State")

		center_data.append(total_lowdins[chosen_atoms[i]]) #labels.append("Center Lowdin in State")
		
		center_data.append(total_lowdins[chosen_atoms[i]] - lowdin_averages[atoms[chosen_atoms[i]]] ) #labels.append("Center Total Lowdin Relative to Element Average")

		center_data.append(np.dot(coords[chosen_atoms[i]],dipole)) #labels.append("Center Dipole Overlap")

		center_data.append(tetrahedral_errors[i][0])#labels.append("Tetrahedral Absolute Error")

		center_data.append(tetrahedral_errors[i][1])#labels.append("Tetrahedral Squared Error")

		center_data.append(seesaw_errors[i][0]) #labels.append("SeeSaw Absolute Error")

		center_data.append(seesaw_errors[i][1]) #labels.append("SeeSaw Squared Error")

		center_data.append(n_lig[i]) #labels.append("Number of Bound Ligand")

		closest_4=closest_atoms[i][:4]
		closest_angles=[]
		for j in closest_4:
			for k in closest_4:
				if j>k:
					if np.isnan(angles[chosen_atoms[i]][j][k]):
						closest_angles.append(-1)
					else:
						closest_angles.append(angles[chosen_atoms[i]][j][k])
		sorted_closest_6=sorted(closest_angles,reverse=True)
		
		center_data.append(sorted_closest_6[0]) #labels.append('Bond Angle 1') # 6 closest bond angles, largest to smallest
		center_data.append(sorted_closest_6[1]) #labels.append('Bond Angle 2') # 6 closest bond angles, largest to smallest
		center_data.append(sorted_closest_6[2]) #labels.append('Bond Angle 3') # 6 closest bond angles, largest to smallest
		center_data.append(sorted_closest_6[3]) #labels.append('Bond Angle 4') # 6 closest bond angles, largest to smallest
		center_data.append(sorted_closest_6[4]) #labels.append('Bond Angle 5') # 6 closest bond angles, largest to smallest
		center_data.append(sorted_closest_6[5]) #labels.append('Bond Angle 6') # 6 closest bond angles, largest to smallest

		closest_bonds=[]
		for j in closest_4:
			closest_bonds.append(dists[chosen_atoms[i]][j])
		sorted_4_bonds=sorted(closest_bonds)


		center_data.append(sorted_4_bonds[0]/bond_avs[atoms[chosen_atoms[i]]][atoms[closest_4[closest_bonds.index(sorted_4_bonds[0])]]]) #labels.append('Relative Bond Length 1')# 4 closest adjusted bond lengths, smallest to largest. Relative to QD average
		center_data.append(sorted_4_bonds[1]/bond_avs[atoms[chosen_atoms[i]]][atoms[closest_4[closest_bonds.index(sorted_4_bonds[1])]]]) #labels.append('Relative Bond Length 2')# 4 closest adjusted bond lengths, smallest to largest. Relative to QD average
		center_data.append(sorted_4_bonds[2]/bond_avs[atoms[chosen_atoms[i]]][atoms[closest_4[closest_bonds.index(sorted_4_bonds[2])]]]) #labels.append('Relative Bond Length 3')# 4 closest adjusted bond lengths, smallest to largest. Relative to QD average
		center_data.append(sorted_4_bonds[3]/bond_avs[atoms[chosen_atoms[i]]][atoms[closest_4[closest_bonds.index(sorted_4_bonds[0])]]]) #labels.append('Relative Bond Length 4')# 4 closest adjusted bond lengths, smallest to largest. Relative to QD average

		reference1=coords[closest_atoms[i][0]] - coords[chosen_atoms[i]]
		reference2=coords[closest_atoms[i][1]] - coords[chosen_atoms[i]]

		for j in range(neighbors):
			center_data.append(atoms[closest_atoms[i][j]]) #labels.append('Neighbor '+str(i)+" Element")

			center_data.append(to_atom_specific_index(atoms,closest_atoms[i][j])) #labels.append('Neighbor '+str(i)+" Element-Index")

			center_data.append(closest_atoms[i][j])#labels.append('Neighbor '+str(i)+" Total Index")

			center_data.append(len(connectivity[closest_atoms[i][j]])) #labels.append('Neighbor '+str(i)+" Coordination Number")

			center_data.append(dists[closest_atoms[i][j]][chosen_atoms[i]]) #labels.append('Neighbor '+str(i)+" Distance to Center")

			center_data.append(orbs_by_atoms[state][closest_atoms[i][j]]) #labels.append('Neighbor '+str(i)+" Alpha")

			center_data.append(mulliken[closest_atoms[i][j]]) #labels.append('Neighbor '+str(i)+" Mulliken Charge")

			center_data.append(mulliken[closest_atoms[i][j]] - mulliken_averages[atoms[closest_atoms[i][j]]]) #labels.append('Neighbor '+str(i)+" Mulliken Relative to Element Average")

			center_data.append(low_per_orb[closest_atoms[i][j]][state])	#labels.append('Neighbor '+str(i)+" Lowdin in State")

			center_data.append(total_lowdins[closest_atoms[i][j]]) #labels.append('Neighbor '+str(i)+" Total Lowdin Charge")

			center_data.append(total_lowdins[closest_atoms[i][j]] - lowdin_averages[atoms[closest_atoms[i][j]]]) #labels.append('Neighbor '+str(i)+" Total Lowdin Charge Relative to Element Average")

			projected=vector_projector(coords[closest_atoms[i][j]]-coords[chosen_atoms[i]],reference1,reference2)
			center_data.append(projected[0])#labels.append('Neighbor '+str(i)+" Projected X Coordinate")
			center_data.append(projected[1])#labels.append('Neighbor '+str(i)+" Projected Y Coordinate")
			center_data.append(projected[2])#labels.append('Neighbor '+str(i)+" Projected Z Coordinate")

			if j>0:
				center_data.append(get_angle(coords,dists,chosen_atoms[i],closest_atoms[i][j],closest_atoms[i][0])) #labels.append('Neighbor '+str(i)+" Angle to Center to Center Bond 1")

			if j>1:
				center_data.append(get_dihedral(coords[closest_atoms[i][j]],coords[closest_atoms[i][1]],coords[chosen_atoms[i]],coords[closest_atoms[i][0]]))	#labels.append('Neighbor '+str(i)+" Dihedral Angle to Center Bond 2 to Center to Center Bond 1")

			for k in range(neighbors):
				if j>k and k>0 and j<angle_cutoff:
					center_data.append(get_angle(coords,dists,chosen_atoms[i],closest_atoms[i][j],closest_atoms[i][k])) #labels.append('Neighbor '+str(i)+" Neighbor "+str(j)+" Angle with Center")


		if len(labels)!=len(center_data):
			raise Exception("You missed something: labels",len(labels),"Data",len(center_data))

		data.append(center_data)


	#now I need to actually write everything to a local excel file

	data=np.array(data)
	df=pd.DataFrame(data,columns=labels)
	df.to_excel(file_name,index=False)
	print("All atom data written to excel file",file_name)


def catalogue_all_atoms(atoms,ind_lig,ind_cat,ind_P,alpha_cat,ipr,orbs_by_atoms,dists,angles,connectivity,nocc,ipr_thresh,alpha_thresh,neighbors,vir_max,occ_min):
	'''
	Collects a ton of data about each non-ligand atom in a structure
	Inputs:
		atoms: np array of strs. the element of each atom. len Natoms
		ind_lig: np array of booleans. tells which atoms are ligands. len Natoms
		ind_cat: np array of booleans. tells which atoms are cations. len Natoms
		ind_P: np array of booleans. tells which atoms are anions. len Natoms
		ipr: np array of floats. participation ratios per MO. len nbas
		orbs_by_atoms: 2D array of floats. alpha in each MO for each atom. NAtoms x nbas (atoms by orbs)
		dists: np array of inter-atom distances (floats), NAtoms x NAtoms
		angles: np array of angles. NAtoms x NAtoms x NAtoms. all nonbonded entries are NaN. first index is the center of the angle, so (a,b,c)!=(b,a,c)=(b,c,a)
		connectivity: np array of graph connectivities (ints), NAtoms x Coordination#
		nocc: int. the number of occupied orbitals
		ipr_thresh: float. an arbitrary parameter (0-1). the fraction of the "max" pr that a state could have and still be considered a bulk state
		alpha_thresh: float. an arbitrary parameter (0-1). what fraction of the max alpha in a state for an atom to have for that state to also belong to that atom
		neighbors: int. total number of nearest neighbors for each atom to include info about. 52 is roughly 3rd-nearest neighbors
		occ_min: int. the number of states from the HOMO that the VBM is (i.e. if 0, the HOMO is the VBM)
		vir_max: int. the number of states from the LUMO that the CBM is (i.e. if 0, the LUMO is the CBM)
	Outputs:
		chosen_states: list of ints. 0-based index of the MO chosen for each chosen atom. len nchosen
		closest_atoms:	2D list of ints. 0-based indices of the closest neighbors of each chosen atom. len nchosen x neighbors
		state_labels:	list of strings. labels for each chosen state (bulk vs trap vs structural trap). len nchosen
		chosen_atoms:	list of ints. 0-based indices of each chosen atom. len nchosen
		tetrahedral_errors: 2D np array of floats. squared and absolute errors in bond angle from tetrahedral. len nchosen x 2
		seesaw_errors: 2D np array of floats. squared and absolute errors in bond angle from see-saw. len nchosen x 2
		n_lig:	list of ints. the number of ligands bound to each chosen atom. len nchosen
	'''

	homo=nocc-1
	n_cat=np.count_nonzero(ind_cat)

	chosen_states=[]
	closest_atoms=[]
	state_labels=[]
	chosen_atoms=[]
	tetrahedral_errors=[]
	seesaw_errors=[]
	n_lig=[]
	print(occ_min)
	for i,atom in enumerate(atoms):
		if not ind_lig[i]:
			chosen_atoms.append(i)
			#determine which state corresponds to the atom
			state_found=False
			orb=0
			qc=0
			while not state_found:
				if ind_cat[i]:
					mo_i=nocc+orb 
				else:
					mo_i=homo-orb


				if mo_i>=nocc and orb>3: #so we skip the quantum confined state
					alpha_cat_i=alpha_cat[mo_i]
					nfrac=n_cat/len(atoms)
					ipr_cutoff=(ipr_thresh*nfrac)/alpha_cat_i
					if ipr[mo_i]>ipr_cutoff:
						qc=qc+1

				mo_alphas=orbs_by_atoms[mo_i]
				max_alpha=np.max(mo_alphas)
				if (mo_alphas[i]>(max_alpha*alpha_thresh)) and not qc==1:
					state_found=True
					chosen_states.append(mo_i)
					#print(mo_i,orb)
				else:
					orb=orb+1

			#label state
			if (ind_cat[i] and orb>=vir_max) or (ind_P[i] and orb>=occ_min):
				state_labels.append("Bulk")
			elif len(connectivity[i])>3:
				state_labels.append("Structural")
			else:
				state_labels.append("Under-coordinated")

			#find the N closest atoms the center
			args=np.argsort(dists[i])
			indices=np.array(range(len(atoms)))
			closest=indices[args][1:neighbors+1]
			closest_atoms.append(closest)

			ligand_count=0
			for bound in connectivity[i]:
				if ind_lig[bound]:
					ligand_count=ligand_count+1
			n_lig.append(ligand_count)

			if len(connectivity[i])!=4:
				tetrahedral_errors.append([-1,-1])
				seesaw_errors.append([-1,-1])
			else:
				abserror=0
				sqrderror=0
				center_angles=[]
				angle_indices=[]
				for j in connectivity[i]:
					for k in connectivity[i]:
						if j>k:
							abserror=abserror+abs(angles[i][j][k]-109.47)
							sqrderror=sqrderror+(angles[i][j][k]-109.47)**2
							center_angles.append(angles[i][j][k])
							angle_indices.append([j,k])
				tetrahedral_errors.append([abserror,sqrderror])
				biggest=max(center_angles)
				main_indices=angle_indices[center_angles.index(biggest)]
				angles_90=[]
				for j,indices in enumerate(angle_indices):
					if (indices[0] not in main_indices) and (indices[1] not in main_indices):
						angle_120=center_angles[j]
					elif (indices[0] not in main_indices) or (indices[1] not in main_indices):
						angles_90.append(center_angles[j])
				abs_seesaw=abs(biggest-180)+abs(angle_120-120)
				sqrd_seesaw=(biggest-180)**2+(angle_120-120)**2
				for angle in angles_90:
					abs_seesaw=abs_seesaw+abs(angle-90)
					sqrd_seesaw=sqrd_seesaw+(angle-90)**2
				seesaw_errors.append([abs_seesaw,sqrd_seesaw])

	tetrahedral_errors=np.array(tetrahedral_errors)
	seesaw_errors=np.array(seesaw_errors)

	return chosen_states, closest_atoms, state_labels, chosen_atoms, tetrahedral_errors, seesaw_errors, n_lig


def band_edge_finder(ipr,alpha_p,atoms,alpha_p_fc,mo_e,alpha_cat,alpha_cat_fc,homo,ipr_thresh,n_p_uc,n_cat_uc):
	'''
	A shorter version of the full band edge finding algorithm employed in my paper that does not include information from orbital localizations
	Inputs:
		ipr: np array of floats. participation ratios per MO. len nbas
		alpha_p: np array of floats. alpha summed over all non-ligand anions in each MO. len nbas
		atoms: np array of strs. the element of each atom. len Natoms
		alpha_p_fc: np array of floats. alpha summed over all under-coordinated non-ligand anion in each MO. len nbas
		mo_e: np array of floats. energy (eV) of each MO. len nbas
		alpha_cat: np array of floats. alpha summed over all cations in each MO. len nbas
		alpha_cat_fc: np array of floats. alpha summed over all under-coordinated cations in each MO. len nbas
		homo: the 0-based index of the homo, int.
		ipr_thresh: float. an arbitrary parameter (0-1). the fraction of the "max" pr that a state could have and still be considered a bulk state
		n_p_uc: int. the number of under-coordinated non-ligand anion in the QD
		n_cat_uc: int. the number of under-coordinated cation in the QD
	Outputs:
		occ_min: int. the number of states from the HOMO that the VBM is (i.e. if 0, the HOMO is the VBM)
		vir_max: int. the number of states from the LUMO that the CBM is (i.e. if 0, the LUMO is the CBM)
		vbm_e: float. the energy (eV) of the VBM
		cbm_e: float. the energy (eV) of the CBM
		qc_state: boolean. true if a qc state was found in the first 3 virtuals
		qc_e: float. if a qc was found, its energy (eV). 0 otherwise
	'''	

	#find n_p, nocc, n_cat, alpha_p_uc, alpha_cat_uc
	n_p=np.count_nonzero(np.logical_or((atoms=='P'), np.logical_or((atoms=='S'),(atoms=='Se'))))
	n_cat=np.count_nonzero(np.logical_or((atoms=='In'), np.logical_or((atoms=='Ga'),np.logical_or((atoms=="Al"),(atoms=="Zn")))))
	nocc=homo+1
	alpha_p_uc=alpha_p-alpha_p_fc
	alpha_cat_uc=alpha_cat-alpha_cat_fc

	#valence band first
	orb=0
	vb_looking=True
	while vb_looking:
		mo_i=homo-orb

		ipr_i=ipr[mo_i]
		alpha_p_i=alpha_p[mo_i]
		nfrac=n_p/len(atoms)
		ipr_cutoff=(ipr_thresh*nfrac)/alpha_p_i #i'm just gonna use the thresholds from the paper
		pdos_cutoff=1.5*(n_p_uc/(n_p-n_p_uc))*alpha_p_fc[mo_i]

		if (ipr_i>=ipr_cutoff) and (alpha_p_uc[mo_i]<=pdos_cutoff):
			occ_min=orb
			vb_looking=False
			vbm_e=mo_e[mo_i]

		orb = orb+1

	#conduction band second
	orb=0
	cb_looking=True
	qc_state=False
	qc_e=0
	while cb_looking:
		mo_i=nocc+orb

		ipr_i=ipr[mo_i]
		alpha_cat_i=alpha_cat[mo_i]
		nfrac=n_cat/len(atoms)
		ipr_cutoff=(ipr_thresh*nfrac)/alpha_cat_i #i'm just gonna use the thresholds from the paper
		pdos_cutoff=1.5*(n_cat_uc/(n_cat-n_cat_uc))*alpha_cat_fc[mo_i]

		if (ipr_i>=ipr_cutoff) and (alpha_cat_uc[mo_i]<=pdos_cutoff):
			if orb>2 or qc_state==True:
					vir_max=orb
					cbm_e=mo_e[mo_i]
					cb_looking=False
			else:
				qc_state=True
				qc_e=mo_e[mo_i]

		orb = orb+1

	return occ_min, vir_max, vbm_e, cbm_e, qc_state, qc_e





if __name__ == "__main__":
	main()