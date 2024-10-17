import numpy as np
from sklearn import kernel_ridge as krr
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn import preprocessing as pp
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import time
import os
import sys
from pyscf import gto, scf, ao2mo, dft
import h5py
import math

def main():
	start=time.time()
	#the general goal here is to train a machine learning model to predict converged Fock matrices from SAD guesses for a specific system
	#our training set consists of some number SAD/Fock pairs from an AIMD run
	#these Fock matrices are currently in the AO basis
	#our ML method (for now) is linear kernel ridge regression
	#This code assumes the following file structure:
	#	three subdirectories, one titled "converged_fock" and one titled "sad_guess" and one titled "xyz_coords"
	#	files in the converged_fock subdirectory are formatted "fock_TIMESTEP.npy" (1-indexed)
	#	files in the sad_guess subdirectory are formatted "sad_TIMESTEP.npy" (1-indexed)
	#	files in the xyz_coords subdirectory are formatted "TIMESTEP.xyz" (1-indexed)
	#we work with only the flattened, upper triangular part of each matrix
	#the code currently assumes the molecule or system of interest is charge neutral with spin 1
	#The script can write the ML generated Fock matrices for various training sizes to a specified directory for further analysis
	#	writes each train size to a different .npy in a directory named manually
	#	fock matrices are written in 1D upper triangular form
	#	But it is also capable of doing said analysis internally

	# This implementation will also allow for different splittings of the Hamiltonian, eventually

	dir_name=sys.argv[1] #the name of the directory to write the predicted fock matrices to 

	#user parameters
	train_sizes=[50, 100, 250, 500, 750, 1000, 1500, 2000]
	scaling="none" #options are none, with_std, no_std
	alpha=1 #regularization strength
	kernal='linear' #type of kernel to use
	solve=750 #whether or not to solve the generated Hamiltonians to find total energies, MO energies, etc. Options are all, False, or a specific train size
	save=False #whether or not to save the generated Hamiltonians

	#first, read in the matrices
	_, _, converged_files=next(os.walk("converged_fock"))
	_, _, sad_files=next(os.walk("sad_guess"))
	_, _, xyz_files=next(os.walk("xyz_coords"))
	if len(converged_files)!=len(sad_files): #or len(converged_files)!=len(xyz_files):
		raise Exception("Need equal # of sad/converged/xyz points")

	converged_files=np.sort(converged_files)
	sad_files=np.sort(sad_files)
	xyz_files=np.sort(xyz_files)
	key=[int(fock[5:-4]) for fock in converged_files]
	sort_key=np.argsort(key)
	converged_files=np.take_along_axis(converged_files,sort_key,-1)
	sad_files=np.take_along_axis(sad_files,sort_key,-1)
	xyz_files=np.take_along_axis(xyz_files,sort_key,-1)

	#load just the upper triangilar part of each matrix
	sads=np.array([np.load("sad_guess/"+file)[np.triu_indices(len(np.load("sad_guess/"+file)))] for file in sad_files]) 
	convergeds=np.array([np.load("converged_fock/"+file)[np.triu_indices(len(np.load("converged_fock/"+file)))] for file in converged_files])
	if solve:
		convergeds_full=np.array([np.load("converged_fock/"+file) for file in converged_files])

	if not os.path.exists(dir_name):
		os.mkdir(dir_name)

	all_predicted=loop_krr_fock(train_sizes,sads,convergeds,scaling,dir_name,save)

	if solve:
		if solve=="all":
			mae_tot_e,mae_mo_e=solve_fock_matrices(all_predicted,convergeds_full,xyz_files,train_sizes)
		else:
			try:
				chosen_train=int(solve)
				chosen_ind=train_sizes.index(chosen_train)
				mae_tot_e,mae_mo_e=solve_fock_matrices(all_predicted[chosen_ind],convergeds_full,xyz_files,chosen_train)
			except ValueError:
				print()
				print("Solving skipped: invalid value provided for 'solve' - should be a training size")
				print()

	end=time.time()
	print("Timing:",round(end-start,3),"s")

def solve_fock_matrices(all_predicted,convergeds_full,xyz_files,train_sizes):
	'''
	2 modes: either a for loop over all training sizes of just for a single training size
	Determined by whether train_sizes is a list or an integer
	'''
	if isinstance(train_sizes,int):
		mae_tot_e,mae_mo_e=solve_focks(all_predicted,convergeds_full,xyz_files,train_sizes)
	else:
		mae_tot_e=[]
		mae_mo_e=[]
		for i,train_size in enumerate(train_sizes):
			chosen_ind=train_sizes.index(chosen_train)
			mae_tot_e_i,mae_mo_e_i=solve_focks(all_predicted[chosen_ind],convergeds_full,xyz_files,train_size)
			mae_tot_e.append(mae_tot_e_i)
			mae_mo_e.append(mae_mo_e_i)

	return mae_tot_e,mae_mo_e

def unfold_1d_upper_triangular(predicted_mats,mat_size):

	#need to un-fold the 1D upper triangular predicted Fock matrices
	unfolded_mats=[]
	for j,mat in enumerate(predicted_mats):

		temp_mat=np.zeros((mat_size,mat_size))
		row=0
		col=0
		for i,element in enumerate(mat):

			temp_mat[row][col]=element
			temp_mat[col][row]=element


			col=col+1
			if col==mat_size:
				row=row+1
				col=row
			
		unfolded_mats.append(temp_mat)

	return np.array(unfolded_mats)


def solve_focks(predicted_mats,convergeds_full,xyz_files,train_size):
	test_set=convergeds_full[train_size:]
	xyz_test=xyz_files[train_size:]

	matrix_size=len(convergeds_full[0])
	unfolded_mats=unfold_1d_upper_triangular(predicted_mats,matrix_size)

	tot_e_maes=[]
	mo_e_maes=[]
	for i,predict in enumerate(unfolded_mats):
		xyz_name="xyz_coords/"+xyz_test[i]
		mol=gto.M(atom=xyz_name, basis="def2svp")	

		mol.spin=0
		mol.charge=0
		mol.verbose=4

		reference=test_set[i]
		mf_true = dft.RKS(mol,xc="PBE")
		mf_true.max_cycle=1
		
		print()
		print("Solving Exact Fock")
		print()
		mf_true.get_fock = lambda *args: reference
		mf_true.kernel()
		total_e_true=mf_true.e_tot
		mo_es_true=mf_true.mo_energy


		print()
		print("Solving Predicted Fock")
		print()

		mf_predict= dft.RKS(mol,xc="PBE")
		mf_predict.max_cycle=1
		mf_predict.get_fock = lambda *args: predict
		mf_predict.kernel()
		total_e_predict=mf_predict.e_tot
		mo_es_predict=mf_predict.mo_energy

		tot_e_maes.append(abs(total_e_true-total_e_predict))
		mo_e_maes.append(metrics.mean_absolute_error(mo_es_predict,mo_es_true))

		print()
		print("Statistics at step",i)
		print("Total energy MAE:",sum(tot_e_maes)/len(tot_e_maes))
		print("MO energy MAE:",sum(mo_e_maes)/len(mo_e_maes))
		print()

	return sum(tot_e_maes)/len(tot_e_maes), sum(mo_e_maes)/len(mo_e_maes)

def loop_krr_fock(train_sizes,sads,convergeds,scaling,dir_name,save):

	all_predicted=[]

	for train_size in train_sizes:
		if train_size<len(sads):

			predicted=krr_fock(train_size,sads,convergeds,scaling,dir_name,save)
			all_predicted.append(predicted)

	return all_predicted

def krr_fock(train_size,sads,convergeds,scaling,dir_name,save):

	X_train=sads[:train_size]
	X_test=sads[train_size:]
	y_train=convergeds[:train_size]
	y_test=convergeds[train_size:]

	if scaling=="with_std":
		pipe = make_pipeline(pp.StandardScaler(), krr.KernelRidge(alpha=alpha,kernel=kernel))
	elif scaling=="no_std":
		pipe = make_pipeline(pp.StandardScaler(with_std=False), krr.KernelRidge())
	elif scaling=="none":
		pipe = make_pipeline(krr.KernelRidge())
	else:
		raise Exception("Invalid value of scaling applied")
	
	#where the magic happens			
	pipe.fit(X_train,y_train)
	predicted=pipe.predict(X_test)	

	if save:
		name=dir_name+"/predicted_converged_focks_train_"+str(train_size)+".npy"
		np.save(name,predicted)

	mae_train, mae_test = find_mae(X_train,y_train,predicted,y_test,pipe,train_size)

	return predicted

def find_mae(X_train,y_train,predicted,y_test,pipe,train_size):

	predict_train=pipe.predict(X_train)
	abs_diff_train=np.abs(predict_train-y_train)
	mae_train=np.mean(abs_diff_train)
	print("Train MAE matrix elements w/ train size",train_size,":",np.round(mae_train,7))

	abs_diff_test=np.abs(predicted-y_test)
	mae_test=np.mean(abs_diff_test)
	print("Test MAE matrix elements w/ train size",train_size,":",np.round(mae_test,7))

	return mae_train, mae_test

if __name__ == "__main__":
	main()