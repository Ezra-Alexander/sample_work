import numpy as np
import sys
import matplotlib.pyplot as plt
from qd_helper import *
import copy
from geom_helper import *
import random
from mass_adder import mass_add
from zincblende_vacancy_filler import vacancy_fill

# an updated script for passivating a QD from a charge to neutral
# goes through and either removes a certain number of -1 halogen ligands from 4c surface cations
# or adds a certain number of -1 halogen ligands where best
# avoids the creation of strong dipole moments
# now has a functionality to avoid removing ligands from cations within a certain distance of a set of specified atoms
# this is still probably an ... inelegant implementation

def main():

	xyz = sys.argv[1] #the original dot. In, P, Ga, Al, F, Cl supported
	out = sys.argv[2] #name of .xyz to write
	charge = int(sys.argv[3]) #the total charge of the dot, i.e. the total number of ligands to add/remove. Include negative sign (adding ligands)
	if len(sys.argv) > 4:
		protected_distance = float(sys.argv[4]) #the distance to the following atoms in which to avoid creating 3c cations
		protected_indices = [ int(x) - 1 for x in sys.argv[5:] ] #a list of total indices (don't all have to be the same element)
	else:
		protected_indices = []
		protected_distance = 0

	cutoff = 2.8 #manual parameter for cation-ligand distance. May need to be tweaked
	nncutoff = 3 #We want to avoid making any In-2c. Could be changed to only remove from 5+c in theory

	coords,atoms = read_input_xyz(xyz)

	if charge > 0:
		atoms, coords = remove_ligands(atoms, coords, cutoff, nncutoff, charge, protected_distance, protected_indices)
	elif charge < 0:
		atoms, coords = add_ligands(charge, atoms, coords)
	else:
		raise Exception("Setting charge (variable 3) to 0 tells the code to do nothing. Try a positive number to remove ligands or a negative number to add ligands.")

	write_xyz(out,atoms,coords)

def remove_ligands(atoms, coords, cutoff, nncutoff, charge, protected_distance, protected_indices):

	n_pop = 0
	while n_pop < charge:	

		ind_p = np.logical_or( np.logical_or( atoms == 'P', np.logical_or( atoms == "Se", atoms == "S" )), np.logical_or( atoms == "C", atoms == "Si" )) #all anions
		ind_cat = np.logical_or( atoms == "In", np.logical_or( atoms == "Ga", np.logical_or( atoms == "Al", atoms == "Zn" )))
		ind_lig = np.logical_or( atoms == "F", atoms == "Cl" )
		ind_attach = ind_lig #relic

		#what's undercoordinated?
		cat_underc_ind, p_underc_ind = get_underc_index(coords, ind_cat, ind_p, ind_lig, ind_attach, cutoff, nncutoff, verbose=False)
		n_underc_cat = np.count_nonzero(cat_underc_ind) #the number of in-2c
		n_lig=np.count_nonzero(ind_lig)

		dists=dist_all_points(coords)
		connectivity=connectivity_finder(dists,cutoff)

		target, target_total_index = first_target_remove(n_lig,atoms,connectivity,protected_indices,protected_distance,ind_lig,dists)

		second_target = second_target_remove(atoms,charge,n_pop,ind_lig,dists,target_total_index,connectivity,protected_indices,protected_distance)

		new_coords, new_atoms = remove_coords_atoms(atoms,ind_lig,target,second_target,coords)

		#recount the number of 2c atoms. If it hasn't increased, accept changes
		#ind_p = np.logical_or(new_atoms=='P',np.logical_or(new_atoms=="Se",new_atoms=="S"))
		ind_p = np.logical_or(np.logical_or(new_atoms=='P',np.logical_or(new_atoms=="Se",new_atoms=="S")),np.logical_or(new_atoms=="C",new_atoms=="Si")) #temporary for P to C/Si defects
		ind_cat = np.logical_or(new_atoms=="In",np.logical_or(new_atoms=="Ga", np.logical_or(new_atoms=="Al",new_atoms=="Zn")))
		ind_lig= np.logical_or(new_atoms=="F", new_atoms=="Cl")
		ind_attach = ind_lig #relic

		new_cat_underc_ind,new_p_underc_ind = get_underc_index(new_coords,ind_cat,ind_p,ind_lig,ind_attach,cutoff,nncutoff,verbose=False)
		new_n_underc_cat=np.count_nonzero(new_cat_underc_ind)

		if new_n_underc_cat==n_underc_cat:

			coords=new_coords
			atoms=new_atoms
			if (charge-n_pop)>1:
				n_pop=n_pop+2
			else:
				n_pop=n_pop+1
			print("Removed", n_pop, "ligands")

	return atoms, coords

def add_ligands(charge, atoms, coords):

	n_add=0
	while n_add>charge:

		dipole, coords=approximate_dipole(atoms,coords)
		dists=dist_all_points(coords)
		connect=smart_connectivity_finder(dists,atoms)
		check_3c=smart_connectivity_finder(dists,atoms,flexibility=1.56)
		
		any3c = check_3c_cation(atoms,check_3c)

		if not any3c:
			print("4c add")

			atoms,coords = add_ligand_to_4c(atoms,dipole,coords,connect)

		else:
			print("3c add")
			

			atoms,coords = add_ligand_to_3c(atoms,dipole,coords,check_3c)

		n_add=n_add-1

	return atoms, coords

def add_ligand_to_3c(atoms,dipole,coords,check_3c):

	min_resultant=100
	for i,atom in enumerate(atoms):

		resultant_dipole=np.linalg.norm(dipole-coords[i])

		if resultant_dipole < min_resultant and (atom=="In" or atom=="Ga" or atom=="Al") and len(check_3c[i])<4:

			min_resultant=resultant_dipole
			max_ind=i
		if atom !="In" and atom != "Ga" and atom != "Al" and atom != "P" and atom != "H":
			ligand=atom
	pocket=[max_ind]

	atoms,coords=vacancy_fill(coords,atoms,ligand,pocket,flex=1.5)

	return atoms, coords

def add_ligand_to_4c(atoms,dipole,coords,connect):
	#max_overlap=0
	min_resultant=100
	max_ind=len(atoms)
	for i,atom in enumerate(atoms):
		#overlap=np.dot(dipole,coords[i])
		resultant_dipole=np.linalg.norm(dipole-coords[i])
		#if overlap > max_overlap and (atom=="In" or atom=="Ga" or atom=="Al") and len(connect[i])<5:
		if resultant_dipole < min_resultant and (atom=="In" or atom=="Ga" or atom=="Al") and len(connect[i])<5:
			surface=is_surface(i,atoms,coords)
			if surface:
				#max_overlap=overlap
				min_resultant=resultant_dipole
				max_ind=to_atom_specific_index(atoms,i)
				base_element=atom
		if atom !="In" and atom != "Ga" and atom != "Al" and atom != "P" and atom != "H":
			ligand=atom

	atoms,coords=mass_add(coords,atoms,ligand,base_element,max_ind)

	return atoms,coords

def check_3c_cation(atoms,check_3c):
	any3c=False
	
	for i,atom in enumerate(atoms):
		if len(check_3c[i])<4 and (atom=="In" or atom=="Ga" or atom=="Al"):
			any3c=True
			max_ind=i
			break

	return any3c

def first_target_remove(n_lig,atoms,connectivity,protected_indices,protected_distance,ind_lig,dists):

		distance_check=False
		while not distance_check:

			target=random.randint(1,n_lig) #choose a random target ligand

			count=0
			for i,atom in enumerate(atoms):
				if ind_lig[i]:
					count=count+1
					if count==target:
						target_total_index=i #ligand-specific index transformed to overall index

			safe=True
			for i,bonded_cat in enumerate(connectivity[target_total_index]):
				for j,protected_atom in enumerate(protected_indices):
					if dists[bonded_cat][protected_atom]<protected_distance:
						safe=False

			if safe:
				distance_check=True
				return target, target_total_index

def second_target_remove(atoms,charge,n_pop,ind_lig,dists,target_total_index,connectivity,protected_indices,protected_distance):
		# add a second target that is maximal distance from the first target if you have more than 1 replacement left
		# we make sure this second target is safe here, rather than using the final n-2c count
		second_target=len(atoms)+10 #for initializing purposes
		if (charge-n_pop)>1:
			max_dist=0
			count=0
			for i,atom in enumerate(atoms):
				if ind_lig[i]: #if its a ligand
					count=count+1
					if dists[i][target_total_index]>max_dist: #if it is further away than all ligands checked so far
						safe=True

						for j,anchor in enumerate(connectivity[i]):
							if len(connectivity[anchor])<4: #unless any anchors are less than 4c
								safe = False
							else:
								for k,protected_atom in enumerate(protected_indices): #unless any anchors are close to protected atoms
									if dists[anchor][protected_atom]<protected_distance:
										safe=False
						if safe:
							second_target=count
							max_dist=dists[i][target_total_index]


			return second_target


def remove_coords_atoms(atoms,ind_lig,target,second_target,coords):
	new_coords=[]
	new_atoms=[]
	lig_count=0
	for j,atom in enumerate(atoms): #construct new atoms and coordinates arrays with the two chosen ones missing
		if ind_lig[j]:
			lig_count=lig_count+1
			if lig_count==target or lig_count==second_target:
				pass
			else:
				new_coords.append(coords[j])
				new_atoms.append(atoms[j])
		else:
			new_coords.append(coords[j])
			new_atoms.append(atoms[j])

	new_coords=np.array(new_coords)
	new_atoms=np.array(new_atoms)

	return new_coords, new_atoms


if __name__ == '__main__':
	main()


