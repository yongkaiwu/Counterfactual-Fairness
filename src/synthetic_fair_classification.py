import os
from itertools import product

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from model.dataset import DataSet
from model.loadxml import load_xml_to_cbn
from model.variable import Event
from model.variable import Variable
from synthetic_data_detect import detect_after_remove

src_path = os.path.dirname (os.path.realpath (__file__))
yt_pos = 1
spos = 1
sneg = 0
tau = 0.050 - 0.0005


def method0(acc_matrix):
	"""
	use all information, regardless fairness
	:return:
	"""
	train_df = pd.read_csv (os.path.join (src_path, '../data/synthetic/synthetic_train.txt'), sep='\t')
	test_df = pd.read_csv (os.path.join (src_path, '../data/synthetic/synthetic_test.txt'), sep='\t')

	x = train_df[['A', 'S', 'M', 'N']].values
	y = train_df['Y'].values

	acc = []
	for name, clf in zip (['LR', 'SVM'], [LogisticRegression (penalty='l2', solver='liblinear'), SVC (kernel='poly', gamma='auto')]):
		clf.fit (x, y)
		train_df[name] = clf.predict (train_df[['A', 'S', 'M', 'N']].values)
		test_df[name] = clf.predict (test_df[['A', 'S', 'M', 'N']].values)

		acc.append (accuracy_score (train_df['Y'], train_df[name]))
		acc.append (accuracy_score (test_df['Y'], test_df[name]))

	acc_matrix.iloc[:, 0] = acc
	train_df.to_csv ('temp/synthetic_train_prediction0.csv', index=False)
	test_df.to_csv ('temp/synthetic_test_prediction0.csv', index=False)


def method1(acc_matrix):
	"""
	we only use the non-descendants of S
	:return:
	"""
	train_df = pd.read_csv (os.path.join (src_path, '../data/synthetic/synthetic_train.txt'), sep='\t')
	test_df = pd.read_csv (os.path.join (src_path, '../data/synthetic/synthetic_test.txt'), sep='\t')

	x = train_df[['A']].values
	y = train_df['Y'].values
	acc = []
	for name, clf in zip (['LR', 'SVM'], [LogisticRegression (penalty='l2', solver='liblinear'), SVC (kernel='poly', gamma='auto')]):
		clf.fit (x, y)
		train_df[name] = clf.predict (train_df[['A']].values)
		test_df[name] = clf.predict (test_df[['A']].values)

		acc.append (accuracy_score (train_df['Y'], train_df[name]))
		acc.append (accuracy_score (test_df['Y'], test_df[name]))

	acc_matrix.iloc[:, 1] = acc
	test_df.to_csv ('temp/synthetic_test_prediction1.csv', index=False)


def method2(acc_matrix):
	train_df = pd.read_csv (os.path.join (src_path, '../data/synthetic/synthetic_train.txt'), sep='\t')
	test_df = pd.read_csv (os.path.join (src_path, '../data/synthetic/synthetic_test.txt'), sep='\t')

	# estimate residual error for the descendants
	model_m = LinearRegression ()
	model_m.fit (train_df[['A', 'S']], train_df['M'])
	train_df['Error_M'] = train_df['M'] - model_m.predict (X=train_df[['A', 'S']])
	test_df['Error_M'] = test_df['M'] - model_m.predict (X=test_df[['A', 'S']])

	model_n = LinearRegression ()
	model_n.fit (train_df[['A', 'S']], train_df['N'])
	train_df['Error_N'] = train_df['N'] - model_n.predict (X=train_df[['A', 'S']])
	test_df['Error_N'] = test_df['N'] - model_n.predict (X=test_df[['A', 'S']])

	x = train_df[['A', 'Error_M', 'Error_N']].values
	y = train_df['Y'].values
	acc = []

	# build classifiers using residual errors
	for name, clf in zip (['LR', 'SVM'], [LogisticRegression (penalty='l2', solver='liblinear'), SVC (kernel='poly', gamma='auto')]):
		clf.fit (x, y)
		train_df[name] = clf.predict (train_df[['A', 'Error_M', 'Error_N']].values)
		test_df[name] = clf.predict (test_df[['A', 'Error_M', 'Error_N']].values)

		acc.append (accuracy_score (train_df['Y'], train_df[name]))
		acc.append (accuracy_score (test_df['Y'], test_df[name]))

	acc_matrix.iloc[:, 2] = acc
	test_df.drop (['Error_M', 'Error_N'], axis=1)
	test_df.to_csv ('temp/synthetic_test_prediction2.csv', index=False)


def method3(acc_matrix):
	df_train = pd.read_csv ('temp/synthetic_train_prediction0.csv')
	train = DataSet (df_train)
	df_test = pd.read_csv ('temp/synthetic_test_prediction0.csv')
	test = DataSet (df_test)
	acc = []

	for name in ['LR', 'SVM']:
		probabilistic_cbn = load_xml_to_cbn (os.path.join (src_path, '../data/synthetic/ProbabilisticBayesianModel.xml'))

		def find_condition_prob(e, t):
			return probabilistic_cbn.find_prob (e, t)

		def get_loc(e):
			return probabilistic_cbn.get_loc (e)

		A = probabilistic_cbn.v['A']
		S = probabilistic_cbn.v['S']
		N = probabilistic_cbn.v['N']
		M = probabilistic_cbn.v['M']
		Y = probabilistic_cbn.v['Y']

		YH = Variable (name='YH', index=Y.index + 1, domains=Y.domains)
		probabilistic_cbn.v[(YH.index, YH.name)] = YH

		YT = Variable (name='YT', index=Y.index + 2, domains=Y.domains)
		probabilistic_cbn.v[(YT.index, YT.name)] = YT

		# build linear loss function
		C_vector = np.zeros ((2 ** 6 + 2 ** 6 // 2, 1))
		for a, n, m, s in product (A.domains.get_all (), N.domains.get_all (), M.domains.get_all (), S.domains.get_all ()):
			p_x_s = train.get_marginal_prob (Event ({'A': a, 'M': m, 'N': n, 'S': s}))

			p_yh_1_y = p_x_s * train.count (Event ({'Y': 0, name: 0}), Event ({'A': a, 'M': m, 'N': n, 'S': s}), 'notequal')
			loc = get_loc (Event ({A: a, M: m, N: n, S: s, YH: 0, YT: 0}))
			C_vector[loc] = p_yh_1_y * train.get_conditional_prob (Event ({name: 0}), Event ({'A': a, 'M': m, 'N': n, 'S': s}))
			loc = get_loc (Event ({A: a, M: m, N: n, S: s, YH: 1, YT: 1}))
			C_vector[loc] = p_yh_1_y * train.get_conditional_prob (Event ({name: 1}), Event ({'A': a, 'M': m, 'N': n, 'S': s}))

			p_yh__y = p_x_s * train.count (Event ({'Y': 0, name: 0}), Event ({'A': a, 'M': m, 'N': n, 'S': s}), 'equal')
			loc = get_loc (Event ({A: a, M: m, N: n, S: s, YH: 0, YT: 1}))
			C_vector[loc] = p_yh__y * train.get_conditional_prob (Event ({name: 0}), Event ({'A': a, 'M': m, 'N': n, 'S': s}))
			loc = get_loc (Event ({A: a, M: m, N: n, S: s, YH: 1, YT: 0}))
			C_vector[loc] = p_yh__y * train.get_conditional_prob (Event ({name: 1}), Event ({'A': a, 'M': m, 'N': n, 'S': s}))

		# the inequality of max and min
		G_matrix_1 = np.zeros ((2 ** 6, 2 ** 6 + 2 ** 6 // 2))
		h_1 = np.zeros (2 ** 6)
		# max
		i = 0
		for a, n, s, yt in product (A.domains.get_all (), N.domains.get_all (), S.domains.get_all (), YT.domains.get_all ()):
			for m in M.domains.get_all ():
				for yh in YH.domains.get_all ():
					loc = get_loc (Event ({A: a, M: m, N: n, S: s, YH: yh, YT: yt}))
					G_matrix_1[i, loc] = train.get_conditional_prob (Event ({name: yh}), Event ({'A': a, 'M': m, 'N': n, 'S': s}))
				loc = get_loc (Event ({A: a, N: n, S: s, YT: yt}))
				G_matrix_1[i, 2 ** 6 + loc] = -1
				i += 1
		# min
		assert i == 2 ** 6 // 2
		for a, n, s, yt in product (A.domains.get_all (), N.domains.get_all (), S.domains.get_all (), YT.domains.get_all ()):
			for m in M.domains.get_all ():
				for yh in YH.domains.get_all ():
					loc = get_loc (Event ({A: a, M: m, N: n, S: s, YH: yh, YT: yt}))
					G_matrix_1[i, loc] = -train.get_conditional_prob (Event ({name: yh}), Event ({'A': a, 'M': m, 'N': n, 'S': s}))
				loc = get_loc (Event ({A: a, N: n, S: s, YT: yt}))
				G_matrix_1[i, 2 ** 6 + 2 ** 6 // 4 + loc] = 1
				i += 1

		# build counterfactual fairness constraints
		G_matrix_2 = np.zeros ((2 ** 2 * 2, 2 ** 6 + 2 ** 6 // 2))
		h_2 = np.ones (2 ** 2 * 2) * tau

		i = 0
		for a, m in product (A.domains.get_all (), M.domains.get_all ()):
			for n in N.domains.get_all ():
				loc = get_loc (Event ({A: a, N: n, S: spos, YT: yt_pos}))
				G_matrix_2[i, 2 ** 6 + loc] = find_condition_prob (Event ({N: n}), Event ({A: a, S: spos}))

				for yh in YH.domains.get_all ():
					loc = get_loc (Event ({A: a, M: m, N: n, S: sneg, YH: yh, YT: yt_pos}))
					G_matrix_2[i, loc] = -find_condition_prob (Event ({N: n}), Event ({A: a, S: sneg})) \
										 * train.get_conditional_prob (Event ({name: yh}), Event ({'A': a, 'M': m, 'N': n, 'S': sneg}))
			i += 1

		assert i == 2 ** 2
		for a, m in product (A.domains.get_all (), M.domains.get_all ()):
			for n in N.domains.get_all ():
				loc = get_loc (Event ({A: a, N: n, S: spos, YT: yt_pos}))
				G_matrix_2[i, 2 ** 6 + 2 ** 6 // 4 + loc] = -find_condition_prob (Event ({N: n}), Event ({A: a, S: spos}))

				for yh in YH.domains.get_all ():
					loc = get_loc (Event ({A: a, M: m, N: n, S: sneg, YH: yh, YT: yt_pos}))
					G_matrix_2[i, loc] = find_condition_prob (Event ({N: n}), Event ({A: a, S: sneg})) \
										 * train.get_conditional_prob (Event ({name: yh}), Event ({'A': a, 'M': m, 'N': n, 'S': sneg}))
			i += 1

		###########

		# mapping in [0, 1]
		G_matrix_3 = np.zeros (((2 ** 6 + 2 ** 6 // 2) * 2, 2 ** 6 + 2 ** 6 // 2))
		h_3 = np.zeros ((2 ** 6 + 2 ** 6 // 2) * 2)

		for i in range (2 ** 6 + 2 ** 6 // 2):
			G_matrix_3[i, i] = 1
			h_3[i] = 1

			G_matrix_3[2 ** 6 + 2 ** 6 // 2 + i, i] = -1
			h_3[2 ** 6 + 2 ** 6 // 2 + i] = 0

		# sum = 1
		A_matrix = np.zeros ((2 ** 6 // 2, 2 ** 6 + 2 ** 6 // 2))
		b = np.ones (2 ** 6 // 2)

		i = 0
		for a, n, m, s, yh in product (A.domains.get_all (), N.domains.get_all (), M.domains.get_all (), S.domains.get_all (), YH.domains.get_all ()):
			for yt in YT.domains.get_all ():
				A_matrix[i, get_loc (Event ({A: a, M: m, N: n, S: s, YH: yh, YT: yt}))] = 1
			i += 1

		assert i == 2 ** 6 // 2

		# combine the inequality constraints
		G_matrix = np.vstack ([G_matrix_1, G_matrix_2, G_matrix_3])
		h = np.hstack ([h_1, h_2, h_3])

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
		train.df[name + '1'] = train.df[name]
		test.df[name + '1'] = test.df[name]
		for a, n, m, s, yh, yt in product (A.domains.get_all (), N.domains.get_all (), M.domains.get_all (), S.domains.get_all (), YH.domains.get_all (), YT.domains.get_all ()):
			if yh.name != yt.name:
				p = mapping[get_loc (Event ({A: a, M: m, N: n, S: s, YH: yh, YT: yt})), 0]
				train.random_assign (Event ({name: yh, 'A': a, 'M': m, 'N': n, 'S': s}), Event ({name + '1': yt}), p)
				test.random_assign (Event ({name: yh, 'A': a, 'M': m, 'N': n, 'S': s}), Event ({name + '1': yt}), p)

		train.df[name] = train.df[name + '1']
		train.df.drop ([name + '1'], axis=1)
		test.df[name] = test.df[name + '1']
		test.df.drop ([name + '1'], axis=1)
		acc.append (accuracy_score (train.df['Y'], train.df[name]))
		acc.append (accuracy_score (test.df['Y'], test.df[name]))

	acc_matrix.iloc[:, 3] = acc

	train.df.to_csv ('temp/synthetic_train_prediction3.csv', index=False)
	test.df.to_csv ('temp/synthetic_test_prediction3.csv', index=False)


def detect_classifier(ce_matrix):
	cbn = load_xml_to_cbn (os.path.join (src_path, '../data/synthetic/ProbabilisticBayesianModel.xml'))
	A = cbn.v['A']
	S = cbn.v['S']
	N = cbn.v['N']
	M = cbn.v['M']
	Y = cbn.v['Y']

	for i in [0, 1, 2, 3]:  # two datasets generated by two methods
		test = DataSet (pd.read_csv ('temp/synthetic_test_prediction%d.csv' % i))
		for j, label in enumerate (['LR', 'SVM']):  # two classifiers
			# modify cpt of label before detect
			for a, n, m, s, y in product (A.domains.get_all (), N.domains.get_all (), M.domains.get_all (), S.domains.get_all (), Y.domains.get_all ()):
				cbn.set_conditional_prob (Event ({Y: y}), Event ({A: a, M: m, N: n, S: s}),
										  test.get_conditional_prob (Event ({label: y}), Event ({'A': a, 'M': m, 'N': n, 'S': s})))
			cbn.build_joint_table ()

			for k, (aprime, mprime) in enumerate (product ([0, 1], [0, 1])):
				p_u, p_l = detect_after_remove (cbn=cbn, s=spos, sprime=sneg, y=1, aprime=aprime, mprime=mprime)
				p = detect_after_remove (cbn=cbn, s=sneg, sprime=sneg, y=1, aprime=aprime, mprime=mprime)
				ce_matrix.iloc[j * 8 + k, 3 * i:3 * i + 2] = [p_u - p, p_l - p]

			for k, (aprime, mprime) in enumerate (product ([0, 1], [0, 1])):
				p_u, p_l = detect_after_remove (cbn=cbn, s=sneg, sprime=spos, y=1, aprime=aprime, mprime=mprime)
				p = detect_after_remove (cbn=cbn, s=spos, sprime=spos, y=1, aprime=aprime, mprime=mprime)
				ce_matrix.iloc[j * 8 + k + 4, 3 * i:3 * i + 2] = [p_u - p, p_l - p]


def pearl_detect_classifier(ce_matrix):
	cbn = load_xml_to_cbn (os.path.join (src_path, '../data/synthetic/DeterministicBayesianModel.xml'))
	UA = cbn.v['UA']
	UN = cbn.v['UN']
	UM = cbn.v['UM']
	US = cbn.v['US']
	UY = cbn.v['UY']
	A = cbn.v['A']
	S = cbn.v['S']
	N = cbn.v['N']
	M = cbn.v['M']
	Y = cbn.v['Y']

	cbn.build_joint_table ()
	event = cbn.jpt.groupby (Event ({UA: 1, UN: 1, UM: 1, US: 1, A: 1, M: 1, S: 1}).keys ())
	condition = cbn.jpt.groupby (Event ({A: 1, M: 1, S: 1}).keys ())

	def pearl_after_remove_(s, sprime, y, aprime, mprime):
		p = 0.0
		for ua, un, um, us in product (UA.domains.get_all (), UN.domains.get_all (), UM.domains.get_all (), US.domains.get_all ()):
			e = Event ({UA: ua, UN: un, UM: um, US: us, A: aprime, M: mprime, S: sprime})
			c = Event ({A: aprime, M: mprime, S: sprime})
			ps = event.get_group (tuple (e.values ()))['prob'].sum () / condition.get_group (tuple (c.values ()))['prob'].sum ()

			for a, n, m in product (A.domains.get_all (), N.domains.get_all (), M.domains.get_all ()):
				p += cbn.find_prob (Event ({A: a}), Event ({UA: ua})) * \
					 cbn.find_prob (Event ({M: m}), Event ({S: s, A: a, UM: um})) * \
					 cbn.find_prob (Event ({N: n}), Event ({S: s, A: a, UN: un})) * \
					 cbn.find_prob (Event ({Y: y}), Event ({S: s, A: a, N: n, M: m, UY: 1})) * \
					 ps
		return p

	for i in [0, 1, 2, 3]:  # two datasets generated by two methods
		test = DataSet (pd.read_csv ('temp/synthetic_test_prediction%d.csv' % i))
		for j, label in enumerate (['LR', 'SVM']):  # two classifiers
			# modify cpt of label before detect
			for a, n, m, s, y in product (A.domains.get_all (), N.domains.get_all (), M.domains.get_all (), S.domains.get_all (), Y.domains.get_all ()):
				cbn.set_conditional_prob (Event ({Y: y}), Event ({A: a, M: m, N: n, S: s, UY: 1}),
										  test.get_conditional_prob (Event ({label: y}), Event ({'A': a, 'M': m, 'N': n, 'S': s})))

			for k, (aprime, mprime) in enumerate (product ([0, 1], [0, 1])):
				ce = pearl_after_remove_ (s=spos, sprime=sneg, y=1, aprime=aprime, mprime=mprime) - \
					 pearl_after_remove_ (s=sneg, sprime=sneg, y=1, aprime=aprime, mprime=mprime)
				ce_matrix.iloc[j * 8 + k, 3 * i + 2] = ce

			for k, (aprime, mprime) in enumerate (product ([0, 1], [0, 1])):
				ce = pearl_after_remove_ (s=sneg, sprime=spos, y=1, aprime=aprime, mprime=mprime) - \
					 pearl_after_remove_ (s=spos, sprime=spos, y=1, aprime=aprime, mprime=mprime)
				ce_matrix.iloc[j * 8 + k + 4, 3 * i + 2] = ce


if __name__ == '__main__':
	acc_matrix = pd.DataFrame (np.zeros ((4, 4)), columns=['BL', 'A1', 'A3', 'CF'])
	method0 (acc_matrix)
	method1 (acc_matrix)
	method2 (acc_matrix)
	method3 (acc_matrix)
	print (acc_matrix.round (5))

	ce_matrix = pd.DataFrame (data=np.zeros ((4 * 4, 3 * 4)), columns=['|- ub', 'lb', 'truth -|'] * 4)
	detect_classifier (ce_matrix)
	pearl_detect_classifier (ce_matrix)
	pd.set_option ('display.max_columns', 13)
	pd.set_option ('display.max_rows', 16)
	pd.set_option ('max_colwidth', 800)
	pd.set_option ('display.width', 1000)
	ce_matrix = ce_matrix.drop (range (4, 8)).drop (range (12, 16))
	ce_matrix.to_csv ('output/synthetic_fair_classification_result.csv', index=False)
	print (ce_matrix.round (3))
