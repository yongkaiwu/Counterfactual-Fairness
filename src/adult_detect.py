import logging
import os
from itertools import product

import numpy as np
import pandas as pd

from model.loadxml import load_xml_to_cbn
from model.variable import Event

logger = logging.getLogger (__name__)
logging.basicConfig (
	format="[%(filename)s:%(lineno)d - %(funcName)s()]-%(msg)s",
	filename='temp/adult.log',
	level=logging.DEBUG
)
cwd = os.getcwd ()


def compute_from_observed(s, sprime, y, a1prime, m1prime, m2prime, a2prime):
	probabilistic_cbn = load_xml_to_cbn (cwd + '/../data/adult/adult.xml')
	probabilistic_cbn.build_joint_table ()

	A1 = probabilistic_cbn.v['age']
	A2 = probabilistic_cbn.v['education']
	S = probabilistic_cbn.v['sex']
	M1 = probabilistic_cbn.v['workclass']
	M2 = probabilistic_cbn.v['marital-status']
	N = probabilistic_cbn.v['hours']
	Y = probabilistic_cbn.v['income']

	# Let's compute a counterfactual statement that is identifiable
	# print ('-' * 20)

	if s == sprime:
		probabilistic_cbn.build_joint_table ()
		# print ('Identifiable:')
		# print ('Compute according the Bayesian network: ', end=''),
		p = probabilistic_cbn.get_conditional_prob (Event ({Y: y}), Event ({A1: a1prime, A2: a2prime, M1: m1prime, M2: m2prime, S: sprime}))
		return p

	else:
		# print ('Unidentifiable:')
		p_u = 0.0
		p_l = 0.0
		for n in N.domains.get_all ():
			p_max = -1
			p_min = 2
			for m1, m2 in product (M1.domains.get_all (), M2.domains.get_all ()):
				p_m = probabilistic_cbn.find_prob (Event ({Y: y}), Event ({A1: a1prime, A2: a2prime, N: n, M1: m1, M2: m2, S: s}))
				p_max = max (p_m, p_max)
				p_min = min (p_m, p_min)
			# print(p_max, p_min)
			p_n = probabilistic_cbn.find_prob (Event ({N: n}), Event ({A1: a1prime, A2: a2prime, M1: m1prime, M2: m2prime, S: s}))
			p_u += p_max * p_n
			p_l += p_min * p_n

		return p_u, p_l


def detect_after_remove(cbn, s, sprime, y, a1prime, m1prime, m2prime, a2prime):
	A1 = cbn.v['age']
	A2 = cbn.v['education']
	S = cbn.v['sex']
	M1 = cbn.v['workclass']
	M2 = cbn.v['marital-status']
	N = cbn.v['hours']
	Y = cbn.v['income']

	if s == sprime:
		cbn.build_joint_table ()
		logger.info ('Identifiable:')
		p = cbn.get_conditional_prob (Event ({Y: y}), Event ({A1: a1prime, A2: a2prime, M1: m1prime, M2: m2prime, S: sprime}))
		logger.info ('Compute according the Bayesian network: %f' % p)
		return p

	else:
		logger.info ('Unidentifiable:')
		p_u = 0.0
		p_l = 0.0
		for n in N.domains.get_all ():
			p_m = []
			for m1, m2 in product (M1.domains.get_all (), M2.domains.get_all ()):
				p_m.append (cbn.find_prob (Event ({Y: y}), Event ({A1: a1prime, A2: a2prime, N: n, M1: m1, M2: m2, S: s})))
			p_max = max (p_m)
			p_min = min (p_m)
			# print(p_max, p_min)
			p_n = cbn.find_prob (Event ({N: n}), Event ({A1: a1prime, A2: a2prime, M1: m1prime, M2: m2prime, S: s}))
			p_u += p_max * p_n
			p_l += p_min * p_n

		logger.info ('Upper bound of counterfactual: %f' % p_u)
		logger.info ('Lower bound of counterfactual: %f' % p_l)
		return p_u, p_l


if __name__ == '__main__':
	tau = 0.05
	spos = 1
	sneg = 0
	y_pos = 1

	# s- -> s+
	ce1 = pd.DataFrame (data=np.zeros ((16, 2)), columns=['ub', 'lb'])
	for i, (a1prime, a2prime, m1prime, m2prime) in enumerate (product ([0, 1], [0, 1], [0, 1], [0, 1])):
		p_u, p_l = compute_from_observed (s=spos, sprime=sneg, y=y_pos, a1prime=a1prime, m1prime=m1prime, m2prime=m2prime, a2prime=a2prime)
		p = compute_from_observed (s=sneg, sprime=sneg, y=y_pos, a1prime=a1prime, m1prime=m1prime, m2prime=m2prime, a2prime=a2prime)
		ce1.iloc[i] = (p_u - p), (p_l - p)

	# s+ -> s-
	ce2 = pd.DataFrame (data=np.zeros ((16, 2)), columns=['ub', 'lb'])
	for i, (a1prime, a2prime, m1prime, m2prime) in enumerate (product ([0, 1], [0, 1], [0, 1], [0, 1])):
		p_u, p_l = compute_from_observed (s=sneg, sprime=spos, y=y_pos, a1prime=a1prime, m1prime=m1prime, m2prime=m2prime, a2prime=a2prime)
		p = compute_from_observed (s=spos, sprime=spos, y=y_pos, a1prime=a1prime, m1prime=m1prime, m2prime=m2prime, a2prime=a2prime)
		ce2.iloc[i] = (p_u - p), (p_l - p)

	tab = pd.concat ([ce1, ce2], axis=1).round (decimals=3)
	tab = tab.ix[:, 0:2]
	tab.to_csv ('output/adult_detect_result.csv', index=False)
	print (tab)
