import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

src_path = os.path.dirname (os.path.realpath (__file__))


def adult():
	adult_df = pd.read_csv (os.path.join (src_path, '../data/adult/adult.csv'))

	adult_df['sex'] = adult_df['sex'].apply (lambda a: 1 if a == "Male" else 0)
	adult_df['age'] = adult_df['age'].apply (lambda a: 1 if a > 37 else 0)
	adult_df['workclass'] = adult_df['workclass'].apply (lambda a: 1 if a != 'Private' else 0)
	adult_df['education'] = adult_df['education-num'].apply (lambda a: 1 if a > 9 else 0)
	adult_df['marital-status'] = adult_df['marital-status'].apply (lambda a: 1 if a == "Married-civ-spouse" else 0)
	adult_df['occupation'] = adult_df['occupation'].apply (lambda a: 1 if a == "Craft-repair" else 0)
	adult_df['relationship'] = adult_df['relationship'].apply (lambda a: 1 if a == "Not-in-family" else 0)
	adult_df['race'] = adult_df['race'].apply (lambda a: 1 if a != "White" else 0)
	adult_df['hours'] = adult_df['hours-per-week'].apply (lambda a: 1 if a > 40 else 0)
	adult_df['native-country'] = adult_df['native-country'].apply (lambda a: 1 if a == "United-States" else 0)
	adult_df['income'] = adult_df['income'].apply (lambda a: 1 if a == ">50K" else 0)

	# extrac 5 columns
	adult_df = adult_df[['age', 'sex', 'workclass', 'education', 'marital-status', 'hours', 'income']]
	# balance the data
	ros = RandomOverSampler (random_state=0)
	X_resampled, y_resampled = ros.fit_sample (
		adult_df[['age', 'sex', 'workclass', 'education', 'marital-status', 'hours']],
		adult_df['income'])
	adult_df = pd.DataFrame (data=np.hstack ([X_resampled, y_resampled.reshape ((y_resampled.__len__ (), 1))]),
							 columns=['age', 'sex', 'workclass', 'education', 'marital-status', 'hours', 'income'])

	adult_df.to_csv (os.path.join (src_path, '../data/adult/adult_binary.csv'), index=False)


def preprocessing():
	df = pd.read_csv (os.path.join (src_path, '../data/adult/adult_binary.csv'))

	# split for 80% and 20%
	np.random.seed (2018)
	msk = np.random.rand (df.__len__ ()) < 0.8
	train = df[msk]
	test = df[~msk]

	train.to_csv (os.path.join (src_path, '../data/adult/adult_binary_train.csv'), index=False)
	test.to_csv (os.path.join (src_path, '../data/adult/adult_binary_test.csv'), index=False)


if __name__ == '__main__':
	adult ()
	preprocessing ()
