import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from pyscf import gto, scf, ao2mo, dft
import h5py
import math
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

def main():
	start=time.time()
	#the general goal here is to train a machine learning model to predict converged Fock matrices from SAD guesses for a specific system
	#our training set consists of some number SAD/Fock pairs from an AIMD run
	#these Fock matrices are currently in the AO basis
	#our ML method is a manual implementation of a kernel ridge regression
		#the chosen kernel (inner product) can be varied
	#This code assumes the following file structure:
	#	three subdirectories, one titled "converged_fock" and one titled "sad_guess" and one titled "xyz_coords"
	#	files in the converged_fock subdirectory are formatted "fock_TIMESTEP.npy" (1-indexed)
	#	files in the sad_guess subdirectory are formatted "sad_TIMESTEP.npy" (1-indexed)
	#	files in the xyz_coords subdirectory are formatted "TIMESTEP.xyz" (1-indexed)
	#we work with only the upper triangular part of each matrix
		#blocks along the diagonal are constrained to symmetry
	#the code currently assumes the molecule or system of interest is charge neutral with spin 1 when computing energies from fock matrices
	#The script can write the ML generated Fock matrices for various training sizes to a specified directory for further analysis
	#	writes each train size to a different .npy in a directory named manually
	#	But it is also capable of doing said analysis internally

	dir_name=sys.argv[1] #the name of the directory to write the predicted fock matrices to 
	alpha = float(sys.argv[2]) #regularization strength
	n_split=int(sys.argv[3]) #number of blocks to split into. Must evenly divide UT matrix

	#user parameters
	train_sizes=[50,100,250,500,750, 1000, 1500, 2000,2500]
	#alpha=0.001 #regularization strength
	solve=False #whether or not to solve the generated Hamiltonians to find total energies, MO energies, etc. Options are all, False, or a specific train size
	save=False #whether or not to save the generated Hamiltonians
	#n_split=10 #number of blocks to split into.  
		#not clear to me yet what the constraints on this are, but for now I'm throwing an exception if it doesn't evenly divide the upper-triangular part of the matrix
	inner_product_type='classic' #only supports the Frobenius inner product right now (np.vdot). The alternative is to introduce an overlap matrix
	scaling=False #standard scaling, applied to both inputs and outputs

	#this implementation only supports the blocked KRR splitting approach
	#I have also currently not implemented the span correction term
	#this implementation also currently only implements the full cross-block approach

	#first, read in the matrices
	_, _, converged_files=next(os.walk("converged_fock"))
	_, _, sad_files=next(os.walk("sad_guess"))
	_, _, xyz_files=next(os.walk("xyz_coords"))
	if len(converged_files)!=len(sad_files): #or len(converged_files)!=len(xyz_files):
		raise Exception("Need equal # of sad/converged/xyz points")

	#sort them so they are all numbered 1-N timesteps
	converged_files=np.sort(converged_files)
	sad_files=np.sort(sad_files)
	xyz_files=np.sort(xyz_files)
	key=[int(fock[5:-4]) for fock in converged_files]
	sort_key=np.argsort(key)
	converged_files=np.take_along_axis(converged_files,sort_key,-1)
	sad_files=np.take_along_axis(sad_files,sort_key,-1)
	xyz_files=np.take_along_axis(xyz_files,sort_key,-1)
	n_timesteps = len(xyz_files)

	#load the matrices
	sads_full=np.array([np.load("sad_guess/"+file) for file in sad_files],dtype=np.float128)
	convergeds_full=np.array([np.load("converged_fock/"+file) for file in converged_files],dtype=np.float128)

	#return the UT blocks of each matrix
	ut_block_sads=ut_block(sads_full,n_split)
	ut_block_convergeds=ut_block(convergeds_full,n_split)

	#convert both SADs and SCFs to supermatrices
	sad_super = to_supermat(ut_block_sads)
	scf_super = to_supermat(ut_block_convergeds)

	#make the directory to write outputs
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)

	#all krr happens here
	all_predicted=loop_krr_fock(train_sizes,sad_super,scf_super,n_timesteps,dir_name,save,alpha,inner_product_type,n_split,scaling)

	#solve predicted hamiltonians for total energy / mo energies
	if solve:

		unfolded_all_predicted = []

		for supermat in all_predicted:
			ut_blocks = un_supermat(supermat)
			fullmat = unblock_ut_mats(ut_blocks)
			unfolded_all_predicted.append(fullmat)

		if solve=="all":
			mae_tot_e,mae_mo_e=solve_fock_matrices(unfolded_all_predicted,convergeds_full,xyz_files,train_sizes)
		else:
			try:
				chosen_train=int(solve)
				chosen_ind=train_sizes.index(chosen_train)
				mae_tot_e,mae_mo_e=solve_fock_matrices(unfolded_all_predicted[chosen_ind],convergeds_full,xyz_files,chosen_train)
			except ValueError:
				print()
				print("Solving skipped: invalid value provided for 'solve' - should be a training size")
				print()

	end=time.time()
	print("Timing:",round(end-start,1),"s")

def un_supermat(supermat,n_timesteps,n_split):
	#the inverse of to_supermat

	_, nb = supermat.shape

	n_row_per_block = math.sqrt(nb)

	timesteps_blocks = np.zeros((n_timesteps,n_split,n_row_per_block,n_row_per_block))

	timestep = 0
	block = 0
	for block in supermat:
		row = 0
		col = 0
		for el in row:
			timesteps_blocks[timestep][block][row][col] = el

			if col+1==n_row_per_block:
				row += 1
				col = 0
			else:
				col += 1
		if block+1 == n_split:
			timestep += 1
			block = 0
		else:
			block += 1

	return timesteps_blocks

def to_supermat(ut_blocks):
	#turns a 4D block series (timesteps x blocks x 2D matrices) to a 2D supermatrix of (timesteps * blocks) x block element

	n_timesteps , n_blocks, n_rows, n_cols = ut_blocks.shape

	supermat = np.zeros((n_timesteps*n_blocks,n_rows*n_cols))

	for timestep, blocks in enumerate(ut_blocks):
		for block_n, block_els in enumerate(blocks):
			for row_n, row in enumerate(block_els):
				for col_n,element in enumerate(row):
					supermat[ (timestep * n_blocks) + block_n , (row_n * n_cols) + col_n] = element

	return supermat

def in_prod(A,B,inner_product_type): 
	if inner_product_type=="classic":
		return np.vdot(A,B)
	else:
		raise Exception("Specified inner product not yet implemented")

def ut_block(matrices,n_split):
	'''
	Outputs one set of blocks per matrix, and the blocks are ouput as an array (instead of as a matrix of matrices)
	raises an exception if the matrices are not square
	Inputs:
		a np array of matrices (a np array (timesteps) of np array (rows) x np array (columns))
	Outputs:
		a np array of ut blocked matrices (np array (timesteps) of np array (blocks) of np array (rows) x np array (columns))
	'''
	matrices=np.array(matrices)
	timesteps=[]
	roots=np.roots([1,1,-2*n_split])
	for matrix in matrices:

		rows,cols=matrix.shape
		if rows!=cols:
			raise Exception("Matrices are not square!")
		
		n_blocks_per_row=[x for x in roots if x>0][0]	
		block_nrow=rows/n_blocks_per_row
		if not block_nrow.is_integer():
			raise Exception("Chosen n_split does not evenly divide matrix")
		block_nrow=int(block_nrow)

		blocks=np.array([np.zeros((block_nrow,block_nrow)) for _ in range(n_split)])

		count=0
		for i in range(0, rows, block_nrow):
			for j in range(i, cols, block_nrow):
				blocks[count]=matrix[i:i+block_nrow,j:j+block_nrow]
				count=count+1

		timesteps.append(blocks)

	return np.array(timesteps)

def unblock_ut_mats(blocks):
	'''
	the inverse of the ut_block function
	Inputs:
		a np array of ut blocked matrices (np array (timesteps) of np array (blocks) of np array (rows) x np array (columns))
	Outputs:
		a np array of matrices (a np array (timesteps) of np array (rows) x np array (columns))
	'''
	blocks=np.array(blocks,dtype=np.float128)
	unblocked=[]
	for timestep in blocks:
		n_blocks=len(timestep)
		roots=np.roots([1,1,-2*n_blocks])
		n_blocks_per_row=[x for x in roots if x>0][0]
		block_size=len(timestep[0])
		big_mat_size=int(n_blocks_per_row*block_size)
		big_mat=np.zeros((big_mat_size,big_mat_size),dtype=np.float128)
		block_row=0
		block_col=0
		for block in timestep:
			#print(block_row,block_col)
			big_mat[block_row : block_row + block_size , block_col : block_col + block_size] = block

			if block_row!=block_col: #off-diagonals
				big_mat[block_col : block_col + block_size , block_row : block_row + block_size] = block.T

			if (block_col/block_size)<n_blocks_per_row-1:
				block_col=block_col+block_size
			else:
				block_row=block_row+block_size
				block_col=block_row

		unblocked.append(big_mat)

	return unblocked

def blocked_krr(F_tilde,f_tilde,F,alpha,inner_product_type):

	#note that this implementation computes all the predicts at once as one supermatrix
	#I'm not yet sure if that is equivalent

	ntrain_nB, nb = F_tilde.shape

	ntest_nB, nb = f_tilde.shape

	#this is an assumption, I'm not sure if this is the right way to do this
	gamma_2 = (alpha**2)*np.identity(ntrain_nB)

	if inner_product_type == "classic":

		#both of these might also be wrong
		#each row of the supermatrices is a flattened block
		#so elementwise block inner products are just dot products of the rows
		#currently I compute every block product, but shouldn't only like blocks be compared?
		S_tilde = np.zeros((ntrain_nB,ntrain_nB))

		for i,row1 in enumerate(F_tilde):
			for j,row2 in enumerate(F_tilde):
				S_tilde[i,j] = np.dot(row1,row2)


		f_tilde_by_F_tilde = np.zeros((ntest_nB,ntrain_nB))

		for i,row1 in enumerate(f_tilde):
			for j,row2 in enumerate(F_tilde):
				f_tilde_by_F_tilde[i,j] = np.dot(row1,row2)

	else:
		raise Exception("Other inner products not implemented")

	predicted = f_tilde_by_F_tilde  @ np.linalg.inv(S_tilde + gamma_2) @ F

	return predicted

def elementwise_blocked_krr(F_tilde,f_tilde,F,alpha,inner_product_type,n_split):
	#replaces the matrix multiplication with good old fashioned element-wise computation
	#I assume slower, the important thing is if this gives the same answer

	ntrain_nB, nb = F_tilde.shape

	ntest_nB, nb = f_tilde.shape

	#this is an assumption, I'm not sure if this is the right way to do this
	gamma_2 = (alpha**2)*np.identity(ntrain_nB)

	if inner_product_type == "classic":

		#both of these might also be wrong
		#each row of the supermatrices is a flattened block
		#so elementwise block inner products are just dot products of the rows
		#currently I compute every block product, but shouldn't only like blocks be compared?
		S_tilde = np.zeros((ntrain_nB,ntrain_nB))

		for i,row1 in enumerate(F_tilde):
			for j,row2 in enumerate(F_tilde):
				S_tilde[i,j] = np.dot(row1,row2)

	else:
		raise Exception("Other inner products not implemented")

	M = np.linalg.inv(S_tilde + gamma_2)

	n_predicts = int(ntest_nB / n_split)

	predicted=np.zeros((ntest_nB, nb))
	for time in range(n_predicts):
		for block in range(n_split):
			for block_el in range(nb):

				time_block = (time * n_split) + block
				
				for block_el_b in range(nb):
					for timeblock_bb in range(ntrain_nB):
						for timeblock_cy in range(ntrain_nB):
							predicted[time_block][block_el] += f_tilde[time_block][block_el_b] * F_tilde[timeblock_bb][block_el_b] * M[timeblock_bb][timeblock_cy] * F[timeblock_cy][block_el]



	return predicted

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

	tot_e_maes=[]
	mo_e_maes=[]
	for i,predict in enumerate(predicted_mats):
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
		print("Total energy MAE:",round_2_sigfigs(sum(tot_e_maes)/len(tot_e_maes),3))
		print("MO energy MAE:",round_2_sigfigs(sum(mo_e_maes)/len(mo_e_maes),3))
		print()

	return sum(tot_e_maes)/len(tot_e_maes), sum(mo_e_maes)/len(mo_e_maes)

def round_2_sigfigs(n,sigfigs):

	return round(n,sigfigs-int(math.floor(math.log10(abs(n)))))

def loop_krr_fock(train_sizes,sad_super,scf_super,n_timesteps,dir_name,save,alpha,inner_product_type,n_split,scaling):
	num_workers = os.cpu_count()
	all_predicted=[]

	for train_size in train_sizes:
		if train_size<n_timesteps:

			predicted=krr_fock(train_size,sad_super,scf_super,dir_name,save,alpha,inner_product_type,n_split,scaling)
			all_predicted.append(predicted)

	#all_predicted = Parallel(n_jobs = num_workers)(delayed(krr_fock)(train_size,sad_super,scf_super,dir_name,save,alpha,inner_product_type,n_split,scaling) for train_size in train_sizes if train_size < n_timesteps)

	return all_predicted

def split_arrays(array,n_split,cross,type): #unused

	ut_length=len(array[0])
	test=ut_length/n_split
	if (test*(test+1))/2 < ut_length:
		raise Exception("Too many segments chosen:",test,"is less than the side length")

	split_array=[[] for _ in range(n_split)]

	for mat in array:
		segments=np.array_split(mat,n_split)

		if cross=="more_timesteps":
			if type=="x":
				for i in range(n_split):
					for j in range(n_split):
						split_array[i].append(segments[j])
			elif type=="y":
				for i in range(n_split):
					for j in range(n_split):
						split_array[i].append(segments[i])

		else:
			for i in range(n_split):
				split_array[i].append(segments[i])

	print(np.array(split_array).shape)	

	return split_array

def split_krr_fock(pipe,X_train,y_train,X_test,cross):

	for i,segment in enumerate(X_train):
		pipe.fit(segment,y_train[i])
		predicted_i=pipe.predict(X_test[i])
		if i>0:
			predicted=[np.concatenate([prev, new], axis=0) for prev, new in zip(predicted, predicted_i)]
		else:
			predicted=predicted_i		

	predicted=np.array(predicted)

	if cross=="more_timesteps":
		n_segments=len(X_test)
		n_timesteps=int(len(predicted)/n_segments)
		len_mat=len(predicted[0])
		#print(n_segments,len(predicted),len_mat)
		reshaped=predicted.reshape(n_timesteps,n_segments,len_mat)
		averaged=reshaped.mean(axis=1)
		predicted=np.copy(averaged)

	return predicted

def krr_fock(train_size,sad_super,scf_super,dir_name,save,alpha,inner_product_type,n_split,scaling):

	if scaling: #note that this scaling treats element a of each block the same, which may not be the best way to do this, but its worth trying
		x_scaler = StandardScaler()
		y_scaler = StandardScaler()
		sad_super = x_scaler.fit_transform(sad_super)
		scf_super = y_scaler.fit_transform(scf_super)

	cutoff_ntnb = train_size*n_split
	X_train=sad_super[:cutoff_ntnb]
	X_test=sad_super[cutoff_ntnb:]
	y_train=scf_super[:cutoff_ntnb]
	y_test=scf_super[cutoff_ntnb:]

	#where the magic happens
	predicted = blocked_krr(X_train,X_test,y_train,alpha,inner_product_type)
	predict_train = blocked_krr(X_train,X_train,y_train,alpha,inner_product_type)
	# predicted = elementwise_blocked_krr(X_train,X_test,y_train,alpha,inner_product_type,n_split)
	# predict_train = elementwise_blocked_krr(X_train,X_train,y_train,alpha,inner_product_type,n_split)
	
	if scaling:
		y_train = y_scaler.inverse_transform(y_train)
		predicted = y_scaler.inverse_transform(predicted)
		y_test = y_scaler.inverse_transform(y_test)
		predict_train = y_scaler.inverse_transform(predict_train)

	mae_train, mae_test = find_mae(y_train,predicted,y_test,predict_train,train_size)	

	if save:
		name=dir_name+"/predicted_converged_focks_train_"+str(train_size)+".npy"
		np.save(name,predicted)

	return predicted

def find_mae(y_train,predicted,y_test,predict_train,train_size):

	
	abs_diff_train=np.abs(predict_train-y_train)
	mae_train=np.mean(abs_diff_train)
	print("Train MAE matrix elements w/ train size",train_size,":",round_2_sigfigs(mae_train,3))

	abs_diff_test=np.abs(predicted-y_test)
	mae_test=np.mean(abs_diff_test)
	print("Test MAE matrix elements w/ train size",train_size,":",round_2_sigfigs(mae_test,3))

	return mae_train, mae_test

if __name__ == "__main__":
	main()