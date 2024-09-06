import numpy as np
from pdos_helper import dos_grid_general,get_alpha,get_ao_ind, get_ind_ao_underc
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import random
import pandas as pd
import sys
import math
import scipy.special
from scipy import stats
from optparse import OptionParser
from difflib import SequenceMatcher
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import mplcursors
import time
start=time.time()

def main():
	# like catalog_dot_parameter, but more general
	# can perform a variety of analysis tasks

	# excel file should have rows corresponding to datapoints and columns corresponding to variables
	# and should have no blank lines

	parser=OptionParser()

	parser.add_option("-f","--file",dest="file", help="The file with the data. Must be specified. Should have rows corresponding to datapoints and columns corresponding to variables. Excel and csv supported", action="store",type="string")
	parser.add_option("-F",'--file2',dest="file2",help="A second file with the same columns to be combined with the main file into one dataframe. Excel and csv supported",action="store",default=False,type="string")

	# parser.add_option("-c","--count",dest="count", default=False,help="Count the number of datapoints with criterion",action="store_true")

	parser.add_option("-s","--scatter",dest="scatter", default=False,help="Plot scatter of X and Y",action="store_true")

	parser.add_option("-b","--box",dest="box", default=False,help="Boxplot of all targeted numerical columns. Constraints can be used to add targeted columns",action="store_true")

	parser.add_option("-H","--hist",dest="hist", default=False,help="Plot a histogram of all selected columns, with specified number of bins. Enter # of bins",action="store")

	parser.add_option("-x","--x_label",dest="x_label", default=False, help="The label of the column you want to target, in quotes. Defaults to first column",action="store",type="string")
	parser.add_option("-y","--y_label",dest="y_label", default=False, help="The label of the second column you want to target, in quotes. Defaults to second column",action="store",type="string")
	parser.add_option("-z","--z_label",dest="z_label", default=False, help="The label of the third column you want to target, in quotes. Defaults to third column",action="store",type="string")

	parser.add_option("-m","--mix",dest="mix", default=False,help="Collapse a specified range of M columns of N datapoints to 1 column of M*N datapoints. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars. New column label is the common substring of first and last labels in range",action="store")

	parser.add_option("-a","--average",dest="average", default=False,help="Add a column that is the average of a specified range of columns. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars. New column label is 'Average ' plus the common substring of first and last labels in range",action="store")

	parser.add_option("-M","--max",dest="maxrange", default=False,help="Add a column that is the max of a specified range of columns. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars. New column label is 'Max ' plus the common substring of first and second labels in range",action="store")

	parser.add_option("-N","--min",dest="minrange", default=False,help="Add a column that is the min of a specified range of columns. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars. New column label is 'Min ' plus the common substring of first and second labels in range",action="store")

	parser.add_option("-o","--one_hot",dest="one_hot",default=False,help="Replace any number of columns of categorical variables with one-hot encodings. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars")

	parser.add_option("-t","--stats",dest="stats",default=False,help="Print basic statistics for specified x and y and z columns",action="store_true")

	parser.add_option("-r","--correlation",dest="correlation", default=False,help="Compute and output significant (> 2/sqrt(N)) spearman correlation coefficients under constraints for (x,y). Numeric only, min_periods 100",action="store_true")

	parser.add_option("-c","--compare",dest="compare", default=False,help="Compute counts for the pairs of entries between two columns. Data is rounded to the first decimal place so floats work",action="store_true")

	parser.add_option("-k","--cluster",dest="cluster",default=False,help="Perform t-SNE analysis on data. Plots against x,y, and z. Input is the columns to include in the TSNE. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars.",action="store")

	parser.add_option("-p","--print_all",dest="print_all",default=False,help="Print specified columns for all points in constrained set. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars.",action="store")

	parser.add_option("-d","--double",dest="double",default=False,help="If set, histograms are done as a labeled histogram of the x variable instead of X separate histograms. The value should be the indexing column.",action="store")

	parser.add_option("-l","--scale",dest="scale",default=False,help="If set, scales the specified columns by the specified amount. First argument is what to scale by, second argument is all specified columns in Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars",nargs=2,action="store")

	parser.add_option("-C","--columns",dest="all_columns",default=False,help="If set, prints the labels for all columns",action="store_true")

	(options,args)=parser.parse_args()
	file = options.file
	file2=options.file2
	# count = options.count
	x_label = options.x_label
	y_label = options.y_label
	z_label = options.z_label
	scatter = options.scatter
	box=options.box
	hist=options.hist
	mix=options.mix
	minrange=options.minrange
	average=options.average
	maxrange=options.maxrange
	one_hot=options.one_hot
	stats=options.stats
	corr=options.correlation
	comp=options.compare
	cluster=options.cluster
	print_all=options.print_all
	double=options.double
	scale=options.scale
	all_columns=options.all_columns

	#parse constraint inputs
	constraints=args #positional arguments:  (column labels, relationship, target values). Column labels and relationships.accepted relationships are =, <, >, >=, <=, !=
	constrained=False
	if len(constraints)>0:
		constrained=True
		if len(constraints)%3!=0:
			raise Exception("Positional arguments should be constraints with alternating column labels, relationships and target values")
		
	constraints=[float(x) if x.replace('.','',1).isdigit() else x for x in constraints]
	nice_constraints=[constraints[i:i+3] for i in range(0,len(constraints),3)]	

	#read in the data
	if file.split('.')[1]=="xlsx":
		data=pd.read_excel(file)
	elif file.split('.')[1]=="csv":
		data=pd.read_csv(file)
	else:
		raise Exception("File type unsupported")
	labels=data.columns.tolist()
	if file2:
		if file2.split('.')[1]=="xlsx":
			data2=pd.read_excel(file2)
		elif file2.split('.')[1]=="csv":
			data2=pd.read_csv(file2)
		else:
			raise Exception("File 2 type unsupported")

		data=pd.concat([data,data2],ignore_index=True)

	if all_columns:
		[print(label) for label in labels]

	#fill in x,y,z if undeclared
	if not x_label:
		x_label=labels[0]
	if not y_label:
		y_label=labels[1]
	if not z_label:
		z_label=labels[2]

	#make custom columns
	if mix:
		data=mix_cols(mix,labels,data)
		
	if average:
		data=average_cols(average,labels,data)
		
	if maxrange:
		data=minmax_cols(maxrange,labels,data,"max")	

	if minrange:
		data=minmax_cols(minrange,labels,data,"min")
		
	if one_hot:
		data=one_hot_dataframe(one_hot,labels,data)

	if scale:
		(scale_by, scale_cols) = scale
		data=scale_columns(data,scale_by,scale_cols,labels)

	#do constraints
	if not constrained:
		target_col=data[[x_label,y_label,z_label]]
		everything_but=data.copy()
	else:
		target_col, everything_but = constrain_dataframe(x_label,y_label,z_label,nice_constraints,data)

	#do analysis

	if stats:
		print()
		print(target_col.describe())
		print()

	if corr:
		cross_correlate(everything_but,x_label)

	if scatter:
		target_col.plot.scatter(x=x_label,y=y_label)
		#plt.xlim(0,6600)
		#plt.ylim(0,6600)
		#plt.plot([0,6600],[0,6600],linestyle="dashed")
		#plt.xticks(range(11),range(11))
		plt.show()

	if box:
		box_col=target_col.select_dtypes(exclude=['int64'])
		box_col.plot.box()
		plt.show()

	if hist:
		plot_hist(target_col,x_label,double,hist)

	if comp:
		compare_columns(target_col,x_label,y_label)

	if print_all:
		print_cols(print_all,everything_but,labels)

	if cluster:
		do_tsne(everything_but,x_label,y_label,z_label,cluster,labels)

	end=time.time()
	print(round(end-start,1),"seconds")

def convert_alphabet(input_string):
		'''
		Let's be clear, there's probably a better way to do this built in somewhere
		The goal is to take an string containing some number of excel alphabetical column labels separated by :'s and convert them to a string of int labels
		I'm going to assume that no alphabetical column label is more than two characters
		'''
		alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

		output_string=""
		second=False
		for i,char in enumerate(input_string):
			if (char == ":" or char.isnumeric()) and second==False:
				output_string=output_string+char
			elif second==False:
				first=char
				second=True
				if i+1==len(input_string):
					output_string=output_string+str(alphabet.index(first)+1)
			elif second==True and char==":":
				output_string=output_string+str(alphabet.index(first)+1)
				output_string=output_string+char
				second=False
			else:
				output_string=output_string+str(alphabet.index(char)+1+(len(alphabet)*(alphabet.index(first)+1)))
				second=False

		return output_string

def mix_cols(mix,labels,data):
	'''
	Adds a new column to the dataframe that is a concatenation of several other columns and adjusts the length of the dataframe accordingly
	New column label is the common substring of first and last labels in range
	Inputs:
		mix: str. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars.
		labels: list of strs. dataframe column names
		data: dataframe of interest
	Outputs:
		new dataframe with mixed column and without original columns
	'''
	mix=convert_alphabet(mix)

	split=mix.split(':')
	bounds=[]
	skip=False
	for i,char in enumerate(split):
		if char!="":
			if skip==False:
				bounds.append([int(char),int(split[i+1])])
				skip=True
			else:
				skip=False

	dataframes=[]

	match=SequenceMatcher(None, labels[bounds[0][0]], labels[bounds[-1][-1]]).find_longest_match()
	common_name=labels[bounds[0][0]][match.a:match.a + match.size]
	if common_name[-1]==" ":
		common_name=common_name[:-1]
	print("New mixed column is", common_name)

	labels2mix=[]
	for bound in bounds:
		add_index=bound[0]
		while add_index<=bound[1]:
			labels2mix.append(labels[add_index-1])
			add_index=add_index+1

	for lab in labels2mix:
		df=pd.DataFrame({'A' : []})
		for drop_lab in labels2mix:
			if lab!=drop_lab:
				if df.empty:
					df=data.drop(columns=[drop_lab])
				else:
					df=df.drop(columns=[drop_lab])
		df=df.rename({lab: common_name},axis='columns')
		dataframes.append(df)

	return pd.concat(dataframes)

def average_cols(average,labels,data):
	'''
	Adds a new column to the dataframe that is a row-wise average of several other columns
	New column label is "Average" plus the common substring of first and last labels in range
	Inputs:
		average: str. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars.
		labels: list of strs. dataframe column names
		data: dataframe of interest
	Outputs:
		new dataframe with averaged column and with original columns
	'''
	average=convert_alphabet(average)
	split=average.split(':')
	bounds=[]
	skip=False
	for i,char in enumerate(split):
		if char!="":
			if skip==False:
				bounds.append([int(char),int(split[i+1])])
				skip=True
			else:
				skip=False

	match=SequenceMatcher(None, labels[bounds[0][0]], labels[bounds[-1][1]]).find_longest_match()
	common_name="Average "+labels[bounds[0][0]][match.a:match.a + match.size]
	if common_name[-1]==" ":
		common_name=common_name[:-1]
	print("New averaged column is",common_name)

	labels2mix=[]
	for bound in bounds:
		add_index=bound[0]
		while add_index<=bound[1]:
			labels2mix.append(labels[add_index-1])
			add_index=add_index+1

	#data[common_name] = data.loc[:, labels2mix ].mean(axis=1)

	averaged=pd.DataFrame(data.loc[:, labels2mix ].mean(axis=1),columns=[common_name])

	data=pd.concat([data,averaged],axis=1)

	return data

def minmax_cols(minmaxrange,labels,data,minmax):
	'''
	Adds a new column to the dataframe that is a row-wise min or max of several other columns
	New column label is "Min" or "Max" plus the common substring of first and last labels in range
	Inputs:
		minmaxrange: str. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars.
		labels: list of strs. dataframe column names
		data: dataframe of interest
		minmax: str. either "min" or "max"
	Outputs:
		new dataframe with min/max column and with original columns
	'''
	minmaxrange=convert_alphabet(minmaxrange)

	split=minmaxrange.split(':')
	bounds=[]
	skip=False
	for i,char in enumerate(split):
		if char!="":
			if skip==False:
				bounds.append([int(char),int(split[i+1])])
				skip=True
			else:
				skip=False

	match=SequenceMatcher(None, labels[bounds[0][0]], labels[bounds[-1][1]]).find_longest_match()
	if minmax.lower()=="min":
		common_name="Min "+labels[bounds[0][0]][match.a:match.a + match.size]
	elif minmax.lower()=="max":
		common_name="Max "+labels[bounds[0][0]][match.a:match.a + match.size]
	else:
		raise Exception("4th argument should be 'min' or 'max'")

	if common_name[-1]==" ":
		common_name=common_name[:-1]
	print("New column is",common_name)

	labels2mix=[]
	for bound in bounds:
		add_index=bound[0]
		while add_index<=bound[1]:
			labels2mix.append(labels[add_index-1])
			add_index=add_index+1

	if minmax.lower()=="min":
		new=pd.DataFrame(data.loc[:, labels2mix ].min(axis=1),columns=[common_name])
	elif minmax.lower()=="max":
		new=pd.DataFrame(data.loc[:, labels2mix ].max(axis=1),columns=[common_name])
	
	data=pd.concat([data,new],axis=1)

	return data

def one_hot_dataframe(one_hot,labels,data):
	'''
	Adds new columns to the dataframe that are one-hot encodings several other columns
	Inputs:
		one_hot: str. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars.
		labels: list of strs. dataframe column names
		data: dataframe of interest
		minmax: str. either "min" or "max"
	Outputs:
		new dataframe with encoded+untouched columns and without original columns
	'''
	one_hot=convert_alphabet(one_hot)

	split=one_hot.split(':')
	bounds=[]
	skip=False
	for i,char in enumerate(split):
		if char!="":
			if skip==False:
				bounds.append([int(char),int(split[i+1])])
				skip=True
			else:
				skip=False
	labels2mix=[]
	for bound in bounds:
		add_index=bound[0]
		while add_index<=bound[1]:
			labels2mix.append(labels[add_index-1])
			add_index=add_index+1

	return pd.get_dummies(data,columns=labels2mix)	


def constrain_dataframe(x_label,y_label,z_label,nice_constraints,data):
	'''
	Takes in a dataframe and a list of constraints and returns two dataframes
	One is the entire dataframe under the constraints
	And the other is just 3 specified columns under constraints (and the constrained columns)
	Inputs:
		x/y/z_label: strs. column names to include in the limited dataframe
		nice_constraints: list of len 3 lists. sublists have structure [column_name, "comparison_operator",value]
		data: dataframe of interest
	Outputs:
		target_col: a constrained dataframe consisting of just the xyz and constraint columns
		everything_but: the full dataframe under the constraints
	'''
	columns=[x_label,y_label,z_label]
	for column in nice_constraints:
		if column[0] not in columns:
			columns.append(column[0])
	target_col=data[columns] #target_col is just the specified x and y and z columns + constraints
	everything_but=data.copy() #everything but is every column under constraints 
	for column in nice_constraints:
		if column[1]=="=":
			target_col=target_col[target_col[column[0]] == column[2]]
			everything_but=everything_but[everything_but[column[0]] == column[2]]
		elif column[1]==">":
			target_col=target_col[target_col[column[0]] > column[2]]
			everything_but=everything_but[everything_but[column[0]] > column[2]]
		elif column[1]=="<":
			target_col=target_col[target_col[column[0]] < column[2]]
			everything_but=everything_but[everything_but[column[0]] < column[2]]
		elif column[1]==">=":
			target_col=target_col[target_col[column[0]] >= column[2]]
			everything_but=everything_but[everything_but[column[0]] >= column[2]]
		elif column[1]=="<=":
			target_col=target_col[target_col[column[0]] <= column[2]]
			everything_but=everything_but[everything_but[column[0]] <= column[2]]
		elif column[1]=="!=":
			target_col=target_col[target_col[column[0]] != column[2]]
			everything_but=everything_but[everything_but[column[0]] != column[2]]
		else:
			raise Exception("Accepted relationships are =, <, >, >=, <=, !=")

	return target_col, everything_but


def cross_correlate(everything_but,x_label):
	'''
	Computes spearman correlation coefficients between every float column in the dataframe
	And prints 'significant' ones at 99.999% confidence involving the x_label to the terminal, sorted
	'''
	encoded_objects=pd.get_dummies(everything_but.select_dtypes(include=['object']))
	everything_but=pd.concat([everything_but,encoded_objects],axis=1)
	
	target_data=everything_but[x_label]
	other_columns=everything_but.drop(columns=[x_label])
	spearman_corrs=other_columns.corrwith(target_data,method='spearman')
	spearman_corrs=spearman_corrs.abs().sort_values(3)

	length=len(everything_but)
	critical=stats.t.ppf(0.999995,length-2) #two-tailed

	for index,entry in spearman_corrs.items():
		test=(entry*math.sqrt(length-2))/(math.sqrt(1-(entry**2)))
		if abs(test)>critical:
			print(x_label,"x",index+": ", round(entry,4),"(significance",round(abs(test),2),")")


def plot_hist(target_col,x_label,double,hist):
	'''
	Plot a histogram from a pandas dataframe in one of two ways
	if double:
		a stacked histogram of x_label split by a 2nd column double
	else:
		overlaid histograms of each target float column

	hist is the number of bins
	'''
	if double:
			target_col=target_col[[x_label,double]]
			if target_col[double].dtypes!="object":
				target_col[double]=target_col[double].round(1)
			plt.figure()
			sns.histplot(data=target_col,x=x_label,hue=double,multiple='stack',bins=int(hist),palette='tab10')
			plt.show()
	else:
			hist_cols=target_col.select_dtypes(exclude=['int64','object'])
			for hist_col in hist_cols.columns.tolist():
				plt.hist(hist_cols[hist_col],bins=int(hist),alpha=(1/len(hist_cols.columns.tolist())),label=hist_col)
			plt.legend()
			plt.show()

def compare_columns(target_col,x_label,y_label):
	'''
	Compute counts for the pairs of entries between two columns. Data is rounded to the first decimal place so floats work
	'''
	pairs=[]
	counts={}
	x_float=True
	# print(target_col)
	# print(target_col[y_label].dtypes)
	if target_col[x_label].dtypes == "object":
		x_float=False
	y_float=True
	if target_col[y_label].dtypes == "object":
		y_float=False
	for index,row in target_col.iterrows():
		if x_float:
			x_i=round(row[x_label],2)
		else:
			x_i=row[x_label]
		if y_float:
			y_i=round(row[y_label],2)
		else:
			y_i=row[y_label]
		list_pair=[x_i,y_i]
		tuple_pair=tuple(list_pair)
		if list_pair not in pairs:
			counts[tuple_pair]=0
			pairs.append(list_pair)
		counts[tuple_pair]=counts[tuple_pair]+1

	if x_float:
		pairs.sort(key = lambda x: x[0])
	if y_float:
		pairs.sort(key = lambda x: x[1])

	print()
	print("Out of a total of", len(target_col),"datapoints")
	count=0
	for pair in pairs:
		print("There are",counts[tuple(pair)],"datapoints where",x_label,"is",pair[0],"and",y_label,"is",pair[1])
		if counts[tuple(pair)] > 1:
		 	count=count+counts[tuple(pair)]-1
	print()
	print("Number of redundant rows:",count)
	print()

def print_cols(print_all,everything_but,labels):
	'''
	Print specified columns for all points in constrained set. Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars.
	'''
	print_all=convert_alphabet(print_all)

	split=print_all.split(':')
	bounds=[]
	skip=False
	for i,char in enumerate(split):
		if char!="":
			if skip==False:
				bounds.append([int(char),int(split[i+1])])
				skip=True
			else:
				skip=False

	labels2print=[]
	for bound in bounds:
		add_index=bound[0]
		while add_index<=bound[1]:
			labels2print.append(labels[add_index-1])
			add_index=add_index+1
	printing_data = everything_but.loc[:, labels2print]
	print(labels2print)
	print(len(printing_data))
	# for column in labels2print:
	# 	print()
	# 	print(column)
	# 	print()
	for index, row in printing_data.iterrows():
		for i,column in enumerate(labels2print):
			print(row[column]),
			if i+1==len(labels2print):
				print()

def do_tsne(everything_but,x_label,y_label,z_label,cluster,labels,perplexity=30):
	'''
	Perform t-SNE analysis on data. Plots against x,y, and z. Input is the columns to include in the TSNE. 
	Format col_low:col_high:: ... ::col_low:col_high for noncontiguous ranges as ints or excel chars.
	'''
	cluster=convert_alphabet(cluster)

	split=cluster.split(':')
	bounds=[]
	skip=False
	for i,char in enumerate(split):
		if char!="":
			if skip==False:
				bounds.append([int(char),int(split[i+1])])
				skip=True
			else:
				skip=False

	labels2mix=[]
	for bound in bounds:
		add_index=bound[0]
		while add_index<=bound[1]:
			labels2mix.append(labels[add_index-1])
			add_index=add_index+1

	everything_but_chosen=everything_but[labels2mix]

	#drop all columns with NaN values
	everything_but_chosen=everything_but_chosen.drop(columns=everything_but_chosen.columns[everything_but_chosen.isna().any()].tolist())

	print(everything_but_chosen.shape)

	cat_cols=[]
	num_cols=[]
	for column in everything_but_chosen.columns:
		if everything_but_chosen[column].dtypes=='object':
			cat_cols.append(column)
			print(column)
		else:
			num_cols.append(column)
	encode=OneHotEncoder()
	scale=StandardScaler()
	pipeline=ColumnTransformer(transformers=[('cat',encode,cat_cols),('num',scale,num_cols)])
	pipeline.fit(everything_but_chosen)
	transformed_data=pipeline.transform(everything_but_chosen)
	
	tsne=TSNE(n_components=2,init="pca",learning_rate="auto",perplexity=perplexity).fit_transform(transformed_data)

	everything_but['tsne-2d-one'] = tsne[:,0]
	everything_but['tsne-2d-two'] = tsne[:,1]

	#round x, y, and z to have ~10 different values
	if everything_but[x_label].dtypes=="float64":
		everything_but=round_to_n_distinct(everything_but,x_label,10)
	if everything_but[y_label].dtypes=="float64":
		everything_but=round_to_n_distinct(everything_but,y_label,10)
	if everything_but[z_label].dtypes=="float64":
		everything_but=round_to_n_distinct(everything_but,z_label,10)

	plt.figure(figsize=(16,10))
	sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",hue=x_label,palette=sns.color_palette("hls", len(set(everything_but[x_label]))),data=everything_but,legend="full",alpha=0.5)
	mplcursors.cursor(hover=True).connect("add",on_plot_hover)
	plt.show()
	plt.figure(figsize=(16,10))
	sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",hue=y_label,palette=sns.color_palette("hls", len(set(everything_but[y_label]))),data=everything_but,legend="full",alpha=0.5)
	mplcursors.cursor(hover=True).connect("add",on_plot_hover)
	plt.show()
	plt.figure(figsize=(16,10))
	sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",hue=z_label,palette=sns.color_palette("hls", len(set(everything_but[z_label]))),data=everything_but,legend="full",alpha=0.5)
	mplcursors.cursor(hover=True).connect("add",on_plot_hover)
	plt.show()

def scale_columns(data,scale_by,scale_cols,labels):
	scale_cols=convert_alphabet(scale_cols)

	split=scale_cols.split(':')
	bounds=[]
	skip=False
	for i,char in enumerate(split):
		if char!="":
			if skip==False:
				bounds.append([int(char),int(split[i+1])])
				skip=True
			else:
				skip=False

	labels2mix=[]
	for bound in bounds:
		add_index=bound[0]
		while add_index<=bound[1]:
			labels2mix.append(labels[add_index-1])
			add_index=add_index+1

	for label in labels2mix:
		data[label]*= float(scale_by)
		print(data[label])
	return data

def round_to_n_distinct(everything_but,x_label,n_distinct,error_margin=0.5):
	'''
	Round the values of a dataframe until there are close to n_distinct values
	Inputs:
		everything_but: the dataframe
		x_label: the column of the dataframe to round
		n_distinct: the number of distinct values you want
		error_margin: float. the range n_distinct*margin to ndistinct/margin is considered acceptable
	Outputs:
		the dataframe with the rounded value:
	'''
	current_distinct=everything_but[x_label].nunique()
	max_value = everything_but[x_label].max()
	max_round = -1*(int(np.floor(np.log10(max_value)))+1)
	#print(max_value,max_round,current_distinct)
	current_round=5
	while current_round>=max_round:
		test=everything_but[x_label].round(current_round)
		current_distinct=test.nunique()
		#print(current_distinct,current_round)
		current_round=current_round-1

		if (current_distinct<(n_distinct/error_margin)) and (current_distinct>(n_distinct*error_margin)):
			everything_but[x_label]=test
			return everything_but

	everything_but[x_label]=test	
	return everything_but

def on_plot_hover(sel):
	index=sel.artist.get_offsets()[sel.index]
	datapoint=everything_but.loc[(everything_but['tsne-2d-one'] == index[0]) & (everything_but['tsne-2d-two'] == index[1])]
	print()
	print(datapoint['Dot Name'].values)
	print(datapoint['MO Index'].values)
	print(datapoint['1 or 2 centers?'].values)
	print(round(datapoint['Depth (eV)'].values[0],2))
	print()

if __name__ == "__main__":
	main()



