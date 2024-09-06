import numpy as np
import sys
import pandas as pd
import operator

def main():
	'''
	Takes an .csv and changes values in a column to a specific value if certain conditions are met
	'''
	csv=sys.argv[1] #the file in question
	write_to=sys.argv[2]
	target_col=sys.argv[3] #the column label to target. in quotes
	set_value=sys.argv[4] #what value to set the column to if conditions are met

	constraints=sys.argv[5:]
	if len(constraints)%3!=0:
			raise Exception("Positional arguments should be constraints with alternating column labels, relationships and target values")
		
	constraints=[float(x) if x.replace('.','',1).isdigit() else x for x in constraints]
	nice_constraints=[constraints[i:i+3] for i in range(0,len(constraints),3)]

	data=pd.read_csv(csv)

	fixed_data=set_value_if(data,target_col,set_value,nice_constraints)

	data.to_csv(write_to,index=False)

def set_value_if(data, target_col,set_value,nice_constraints):

	operator_mapping={'=':operator.eq,'!=':operator.ne,'>':operator.gt,'>=':operator.ge,'<':operator.lt,'<=':operator.le}

	for index,row in data.iterrows():

		test = True

		for constraint in nice_constraints:

			constraint_col=constraint[0]
			operator_string=constraint[1]
			constraint_val=constraint[2]

			if not operator_mapping[operator_string](row[constraint_col],constraint_val):
				test=False
				break

		if test:
			data.at[index,target_col]=set_value

	return data

if __name__ == "__main__":
	main()