import os
from itertools import product

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from adult_detect import detect_after_remove
from model.dataset import DataSet
from model.loadxml import load_xml_to_cbn
from model.variable import Event
from model.variable import Variable

src_path = os.path.dirname (os.path.realpath (__file__))
yt_pos = 1
spos = 1
sneg = 0
tau = 0.050 - 0.0010


def method0(acc_matrix):
	"""
	use all information, regardless fairness
	:return:
	"""
	train_df = pd.read_csv (os.path.join (src_path, '../data/adult/adult_binary_train.csv'))
	test_df = pd.read_csv (os.path.join (src_path, '../data/adult/adult_binary_test.csv'))

	x = train_df[['age', 'sex', 'workclass', 'education', 'marital-status', 'hours']].values
	y = train_df['income'].values

	acc = []
	for name, clf in zip (['LR', 'SVM'], [LogisticRegression (penalty='l2', solver='liblinear'), SVC (kernel='poly', gamma='auto')]):
		clf.fit (x, y)
		train_df[name] = clf.predict (train_df[['age', 'sex', 'workclass', 'education', 'marital-status', 'hours']].values)
		test_df[name] = clf.predict (test_df[['age', 'sex', 'workclass', 'education', 'marital-status', 'hours']].values)

		acc.append (accuracy_score (train_df['income'], train_df[name]))
		acc.append (accuracy_score (test_df['income'], test_df[name]))

	acc_matrix.iloc[:, 0] = acc
	train_df.to_csv (os.path.join ('temp/adult_binary_train_prediction0.csv'), index=False)
	test_df.to_csv (os.path.join ('temp/adult_binary_test_prediction0.csv'), index=False)


def method1(acc_matrix):
	"""
	we only use the non-descendants of S
	:return:
	"""
	train_df = pd.read_csv (os.path.join (src_path, '../data/adult/adult_binary_train.csv'))
	test_df = pd.read_csv (os.path.join (src_path, '../data/adult/adult_binary_test.csv'))

	x = train_df[['age', 'education']].values
	y = train_df['income'].values
	acc = []

	for name, clf in zip (['LR', 'SVM'], [LogisticRegression (penalty='l2', solver='liblinear'), SVC (kernel='poly', gamma='auto')]):
		clf.fit (x, y)
		train_df[name] = clf.predict (train_df[['age', 'education']].values)
		test_df[name] = clf.predict (test_df[['age', 'education']].values)

		acc.append (accuracy_score (train_df['income'], train_df[name]))
		acc.append (accuracy_score (test_df['income'], test_df[name]))

	acc_matrix.iloc[:, 1] = acc
	test_df.to_csv ('temp/adult_binary_test_prediction1.csv', index=False)


def method2(acc_matrix):
	train_df = pd.read_csv (os.path.join (src_path, '../data/adult/adult_binary_train.csv'))
	test_df = pd.read_csv (os.path.join (src_path, '../data/adult/adult_binary_test.csv'))
	cbn = load_xml_to_cbn (os.path.join (src_path, '../data/adult/adult.xml'))

	# estimate residual error for the descendants
	for att in ['marital-status', 'workclass', 'hours']:
		att_index = cbn.v.name_dict[att].index
		parents_index = cbn.index_graph.pred[att_index].keys ()
		parents = [cbn.v[i].name for i in parents_index]
		regression = LinearRegression ()
		regression.fit (train_df[parents], train_df[att])

		train_df[att + '-error'] = train_df[att] - regression.predict (train_df[parents])
		test_df[att + '-error'] = test_df[att] - regression.predict (test_df[parents])

	x = train_df[['age', 'education'] + [att + '-error' for att in ['marital-status', 'workclass', 'hours']]].values
	y = train_df['income'].values
	acc = []

	# build classifiers using residual errors
	for name, clf in zip (['LR', 'SVM'], [LogisticRegression (penalty='l2', solver='liblinear'), SVC (kernel='poly', gamma='auto')]):
		clf.fit (x, y)
		train_df[name] = clf.predict (train_df[['age', 'education'] + [att + '-error' for att in ['marital-status', 'workclass', 'hours']]].values)
		test_df[name] = clf.predict (test_df[['age', 'education'] + [att + '-error' for att in ['marital-status', 'workclass', 'hours']]].values)

		acc.append (accuracy_score (train_df['income'], train_df[name]))
		acc.append (accuracy_score (test_df['income'], test_df[name]))

	acc_matrix.iloc[:, 2] = acc
	test_df.drop ([att + '-error' for att in ['marital-status', 'workclass', 'hours']], axis=1)
	test_df.to_csv ('temp/adult_binary_test_prediction2.csv', index=False)


def method3(acc_matrix):
	df_train = pd.read_csv ('temp/adult_binary_train_prediction0.csv')
	# df_train = pd.concat ([df_train] * 10, ignore_index=True)
	train = DataSet (df_train)
	df_test = pd.read_csv ('temp/adult_binary_test_prediction0.csv')
	df_test = pd.concat ([df_test] * 3, ignore_index=True)
	test = DataSet (df_test)
	acc = []

	for name in ['LR', 'SVM']:
		probabilistic_cbn = load_xml_to_cbn (os.path.join (src_path, '../data/adult/adult.xml'))

		def find_condition_prob(e, t):
			return probabilistic_cbn.find_prob (e, t)

		def get_loc(e):
			return probabilistic_cbn.get_loc (e)

		A1 = probabilistic_cbn.v['age']
		A2 = probabilistic_cbn.v['education']
		S = probabilistic_cbn.v['sex']
		M1 = probabilistic_cbn.v['workclass']
		M2 = probabilistic_cbn.v['marital-status']
		N = probabilistic_cbn.v['hours']
		Y = probabilistic_cbn.v['income']

		YH = Variable (name=name, index=Y.index + 1, domains=Y.domains)
		probabilistic_cbn.v[(YH.index, YH.name)] = YH

		YT = Variable (name=name + "M", index=Y.index + 2, domains=Y.domains)
		probabilistic_cbn.v[(YT.index, YT.name)] = YT

		# build linear loss function
		C_vector = np.zeros ((2 ** 8 + 2 ** 8 // 4, 1))
		for a1, a2, n, m1, m2, s in product (A1.domains.get_all (), A2.domains.get_all (), N.domains.get_all (), M1.domains.get_all (), M2.domains.get_all (),
											 S.domains.get_all ()):
			p_x_s = train.get_marginal_prob (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}))

			p_yh_1_y = p_x_s * train.count (Event ({Y: 0, YH: 0}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}), 'notequal')
			loc = get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: 0, YT: 0}))
			C_vector[loc] = p_yh_1_y * train.get_conditional_prob (Event ({YH: 0}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}))
			loc = get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: 1, YT: 1}))
			C_vector[loc] = p_yh_1_y * train.get_conditional_prob (Event ({YH: 1}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}))

			p_yh__y = p_x_s * train.count (Event ({Y: 0, YH: 0}), Event ({A1: a1, A2: a2, M1: m1, M2: m2, N: n, S: s}), 'equal')
			loc = get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: 0, YT: 1}))
			C_vector[loc] = p_yh__y * train.get_conditional_prob (Event ({YH: 0}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}))
			loc = get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: 1, YT: 0}))
			C_vector[loc] = p_yh__y * train.get_conditional_prob (Event ({YH: 1}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}))

		# the inequality of max and min
		G_matrix_1 = np.zeros ((2 ** 8, 2 ** 8 + 2 ** 8 // 4))
		h_1 = np.zeros (2 ** 8)
		# max
		i = 0
		for a1, a2, n, s, yt in product (A1.domains.get_all (), A2.domains.get_all (), N.domains.get_all (), S.domains.get_all (), YT.domains.get_all ()):
			for m1, m2 in product (M1.domains.get_all (), M2.domains.get_all ()):
				for yh in YH.domains.get_all ():
					loc = get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: yh, YT: yt}))
					G_matrix_1[i, loc] = train.get_conditional_prob (Event ({YH: yh}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}))
				loc = get_loc (Event ({A1: a1, A2: a2, N: n, S: s, YT: yt}))
				G_matrix_1[i, 2 ** 8 + loc] = -1
				i += 1
		# min
		assert i == 2 ** 8 // 2
		for a1, a2, n, s, yt in product (A1.domains.get_all (), A2.domains.get_all (), N.domains.get_all (), S.domains.get_all (), YT.domains.get_all ()):
			for m1, m2 in product (M1.domains.get_all (), M2.domains.get_all ()):
				for yh in YH.domains.get_all ():
					loc = get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: yh, YT: yt}))
					G_matrix_1[i, loc] = -train.get_conditional_prob (Event ({YH: yh}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}))
				loc = get_loc (Event ({A1: a1, A2: a2, N: n, S: s, YT: yt}))
				G_matrix_1[i, 2 ** 8 + 2 ** 8 // 8 + loc] = 1
				i += 1

		# build counterfactual fairness constraints
		G_matrix_2 = np.zeros ((2 ** 4 * 2, 2 ** 8 + 2 ** 8 // 4))
		h_2 = np.ones (2 ** 4 * 2) * tau

		i = 0
		for a1, a2, m1, m2 in product (A1.domains.get_all (), A2.domains.get_all (), M1.domains.get_all (), M2.domains.get_all ()):
			for n in N.domains.get_all ():
				loc = get_loc (Event ({A1: a1, A2: a2, N: n, S: spos, YT: yt_pos}))
				G_matrix_2[i, 2 ** 8 + loc] = find_condition_prob (Event ({N: n}), Event ({A1: a1, A2: a2, M1: m1, M2: m2, S: spos}))

				for yh in YH.domains.get_all ():
					loc = get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: sneg, YH: yh, YT: yt_pos}))
					G_matrix_2[i, loc] = -find_condition_prob (Event ({N: n}), Event ({A1: a1, A2: a2, M1: m1, M2: m2, S: sneg})) \
										 * train.get_conditional_prob (Event ({YH: yh}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: sneg}))
			i += 1

		assert i == 2 ** 4
		for a1, a2, m1, m2 in product (A1.domains.get_all (), A2.domains.get_all (), M1.domains.get_all (), M2.domains.get_all ()):
			for n in N.domains.get_all ():
				loc = get_loc (Event ({A1: a1, A2: a2, N: n, S: spos, YT: yt_pos}))
				G_matrix_2[i, 2 ** 8 + 2 ** 8 // 8 + loc] = -find_condition_prob (Event ({N: n}), Event ({A1: a1, A2: a2, M1: m1, M2: m2, S: spos}))

				for yh in YH.domains.get_all ():
					loc = get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: sneg, YH: yh, YT: yt_pos}))
					G_matrix_2[i, loc] = find_condition_prob (Event ({N: n}), Event ({A1: a1, A2: a2, M1: m1, M2: m2, S: sneg})) \
										 * train.get_conditional_prob (Event ({YH: yh}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: sneg}))
			i += 1

		###########

		# mapping in [0, 1]
		G_matrix_3 = np.zeros ((2 * (2 ** 8 + 2 ** 8 // 4), 2 ** 8 + 2 ** 8 // 4))
		h_3 = np.zeros (2 * (2 ** 8 + 2 ** 8 // 4))

		for i in range (2 ** 8 + 2 ** 8 // 4):
			G_matrix_3[i, i] = 1
			h_3[i] = 1

			G_matrix_3[2 ** 8 + 2 ** 8 // 4 + i, i] = -1
			h_3[2 ** 8 + 2 ** 8 // 4 + i] = 0

		# sum = 1
		A_matrix = np.zeros ((2 ** 8 // 2, 2 ** 8 + 2 ** 8 // 4))
		b = np.ones (2 ** 8 // 2)

		i = 0
		for a1, a2, n, m1, m2, s, yh in product (A1.domains.get_all (), A2.domains.get_all (), N.domains.get_all (), M1.domains.get_all (), M2.domains.get_all (),
												 S.domains.get_all (),
												 YH.domains.get_all ()):
			for yt in YT.domains.get_all ():
				A_matrix[i, get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: yh, YT: yt}))] = 1
			i += 1

		assert i == 2 ** 8 // 2

		# combine the inequality constraints
		G_matrix = np.vstack ([G_matrix_1, G_matrix_2, G_matrix_3])
		h = np.hstack ([h_1, h_2, h_3])

		# Test
		# print (np.linalg.matrix_rank (A_matrix), A_matrix.shape[0])
		# print (np.linalg.matrix_rank (np.vstack ([A_matrix, G_matrix])), A_matrix.shape[1])

		# def check():
		# 	sol = np.zeros (2 ** 8 + 2 ** 8 // 4)
		# 	for a1, a2, n, m1, m2, s, yh, yt in product (A1.domains.get_all (), A2.domains.get_all (), N.domains.get_all (), M1.domains.get_all (), M2.domains.get_all (),
		# 												 S.domains.get_all (), YH.domains.get_all (), YT.domains.get_all ()):
		# 		if yh.name == yt.name:
		# 			sol[get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: yh, YT: yt}))] = 1.0
		# 		else:
		# 			sol[get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: yh, YT: yt}))] = 0.0
		#
		# 	for a1, a2, n, s, yt in product (A1.domains.get_all (), A2.domains.get_all (), N.domains.get_all (), S.domains.get_all (), YT.domains.get_all ()):
		# 		p_min = 1
		# 		p_max = 0
		# 		for m1, m2 in product (M1.domains.get_all (), M2.domains.get_all ()):
		# 			p = 0.0
		# 			for yh in YH.domains.get_all ():
		# 				p = train.get_conditional_prob (Event ({YH: yh}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s})) \
		# 					* sol[get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: yh, YT: yt}))]
		# 			if p < p_min:
		# 				p_min = p
		# 			if p > p_max:
		# 				p_max = p
		# 		loc = get_loc (Event ({A1: a1, A2: a2, N: n, S: s, YT: yt}))
		# 		sol[2 ** 8 + loc] = p_max
		# 		sol[2 ** 8 + 2 ** 8 // 8 + loc] = p_min
		#
		# 	np.dot (G_matrix_2, sol)

		# check ()

		# solver
		solvers.options['show_progress'] = False
		sol = solvers.lp (c=matrix (C_vector),
						  G=matrix (G_matrix),
						  h=matrix (h),
						  A=matrix (A_matrix),
						  b=matrix (b),
						  solver=solvers
						  )
		mapping = np.array (sol['x'])

		# build the post-processing result in training and testing
		train.df.loc[:, name + 'M'] = train.df[name]
		test.df[name + 'M'] = test.df[name]
		for a1, a2, n, m1, m2, s, yh, yt in product (A1.domains.get_all (), A2.domains.get_all (), N.domains.get_all (), M1.domains.get_all (), M2.domains.get_all (),
													 S.domains.get_all (), YH.domains.get_all (), YT.domains.get_all ()):
			if yh.name != yt.name:
				p = mapping[get_loc (Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s, YH: yh, YT: yt})), 0]
				train.random_assign (Event ({YH: yh, A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}), Event ({YT: yt}), p)
				test.random_assign (Event ({YH: yh, A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}), Event ({YT: yt}), p)

		train.df[name] = train.df[name + 'M']
		train.df.drop ([name + 'M'], axis=1)
		test.df[name] = test.df[name + 'M']
		test.df.drop ([name + 'M'], axis=1)
		acc.append (accuracy_score (train.df[name], train.df[Y.name]))
		acc.append (accuracy_score (test.df[name], test.df[Y.name]))

	acc_matrix.iloc[:, 3] = acc
	train.df.to_csv ('temp/adult_binary_train_prediction3.csv', index=False)
	test.df.to_csv ('temp/adult_binary_test_prediction3.csv', index=False)


def detect_classifier(ce_matrix):
	cbn = load_xml_to_cbn (os.path.join (src_path, '../data/adult/adult.xml'))

	A1 = cbn.v['age']
	A2 = cbn.v['education']
	S = cbn.v['sex']
	M1 = cbn.v['workclass']
	M2 = cbn.v['marital-status']
	N = cbn.v['hours']
	Y = cbn.v['income']

	for i in [0, 1, 2, 3]:  # two datasets generated by two methods

		test = DataSet (pd.read_csv ('temp/adult_binary_test_prediction%d.csv' % i))
		for j, label in enumerate (['LR', 'SVM']):  # two classifiers

			# modify cpt of label before detect
			for a1, a2, n, m1, m2, s, y in product (A1.domains.get_all (), A2.domains.get_all (), N.domains.get_all (), M1.domains.get_all (), M2.domains.get_all (),
													S.domains.get_all (), Y.domains.get_all ()):
				cbn.set_conditional_prob (Event ({Y: y}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s}),
										  test.get_conditional_prob (Event ({label: y}), Event ({A1: a1, A2: a2, N: n, M1: m1, M2: m2, S: s})))

			cbn.build_joint_table ()
			for k, (a1prime, a2prime, m1prime, m2prime) in enumerate (product ([0, 1], [0, 1], [0, 1], [0, 1])):
				p_u, p_l = detect_after_remove (cbn=cbn, s=spos, sprime=sneg, y=1, a1prime=a1prime, a2prime=a2prime, m1prime=m1prime, m2prime=m2prime)
				p = detect_after_remove (cbn=cbn, s=sneg, sprime=sneg, y=1, a1prime=a1prime, a2prime=a2prime, m1prime=m1prime, m2prime=m2prime)
				ce_matrix.iloc[j * 32 + k, 2 * i:2 * i + 2] = [p_u - p, p_l - p]

			for k, (a1prime, a2prime, m1prime, m2prime) in enumerate (product ([0, 1], [0, 1], [0, 1], [0, 1])):
				p_u, p_l = detect_after_remove (cbn=cbn, s=sneg, sprime=spos, y=1, a1prime=a1prime, a2prime=a2prime, m1prime=m1prime, m2prime=m2prime)
				p = detect_after_remove (cbn=cbn, s=spos, sprime=spos, y=1, a1prime=a1prime, a2prime=a2prime, m1prime=m1prime, m2prime=m2prime)
				ce_matrix.iloc[j * 32 + k + 16, 2 * i:2 * i + 2] = [p_u - p, p_l - p]


if __name__ == '__main__':
	# preprocessing ()

	acc_matrix = pd.DataFrame (np.zeros ((4, 4)), columns=['BL', 'A1', 'A3', 'CF'])
	method0 (acc_matrix)
	method1 (acc_matrix)
	method2 (acc_matrix)
	method3 (acc_matrix)
	print (acc_matrix.round (5))

	ce_matrix = pd.DataFrame (data=np.zeros ((4 * 16, 2 * 4)), columns=['|- ub', 'lb -|'] * 4)
	detect_classifier (ce_matrix)
	pd.set_option ('display.max_columns', 8)
	pd.set_option ('display.max_rows', 64)
	ce_matrix = ce_matrix.drop (range (16, 32)).drop (range (48, 64))
	ce_matrix.to_csv ('output/adult_fair_classification_result.csv', index=False)
	print (ce_matrix.round (3))
