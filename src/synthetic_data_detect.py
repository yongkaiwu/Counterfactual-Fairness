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
	filename='temp/synthetic.log',
	level=logging.DEBUG
)
cwd = os.getcwd ()


def pearl_three_step(s, sprime, y, aprime, mprime):
	deterministic_cbn = load_xml_to_cbn (cwd + '/../data/synthetic/DeterministicBayesianModel.xml')

	UA = deterministic_cbn.v['UA']
	UN = deterministic_cbn.v['UN']
	UM = deterministic_cbn.v['UM']
	US = deterministic_cbn.v['US']
	UY = deterministic_cbn.v['UY']

	A = deterministic_cbn.v['A']
	S = deterministic_cbn.v['S']
	N = deterministic_cbn.v['N']
	M = deterministic_cbn.v['M']
	Y = deterministic_cbn.v['Y']

	"""
	if s == sprime:
		print ('Identifiable:')
		print ("Let's validate pearl's three step by data")
		data = DataSet (pd.read_csv ('../data/synthetic/DeterministicData.txt', sep='\t'))
		print ('Read from data: ', end='')
		print (data.get_conditional_prob (Event ({'Y': y}), Event ({'A': aprime, 'M': mprime, 'S': sprime})))

		p = 0.0
		for ua, un, um, us, uy in product (UA.domains.get_all (), UN.domains.get_all (), UM.domains.get_all (), US.domains.get_all (), UY.domains.get_all ()):
			ps = data.get_conditional_prob (
				Event ({'UA': ua.index, 'UN': un.index, 'UM': um.index, 'US': us.index, 'UY': uy.index}),
				Event ({'A': aprime, 'M': mprime, 'S': sprime}))
			for a, n, m in product (A.domains.get_all (), N.domains.get_all (), M.domains.get_all ()):
				p += deterministic_cbn.find_prob (Event ({A: a}), Event ({UA: ua})) * \
					 deterministic_cbn.find_prob (Event ({M: m}), Event ({S: s, A: a, UM: um})) * \
					 deterministic_cbn.find_prob (Event ({N: n}), Event ({S: s, A: a, UN: un})) * \
					 deterministic_cbn.find_prob (Event ({Y: y}), Event ({S: s, A: a, N: n, M: m, UY: uy})) * \
					 ps
		print ("Pearl's three steps: (U is obtarined from data) %f" % p)
	"""

	p = 0.0
	deterministic_cbn.build_joint_table ()
	for ua, un, um, us, uy in product (UA.domains.get_all (), UN.domains.get_all (), UM.domains.get_all (), US.domains.get_all (), UY.domains.get_all ()):
		# compute p(u|z, s)
		ps = deterministic_cbn.get_conditional_prob (
			Event ({UA: ua.index, UN: un.index, UM: um.index, US: us.index, UY: uy.index}),
			Event ({A: aprime, M: mprime, S: sprime}))

		for a, n, m in product (A.domains.get_all (), N.domains.get_all (), M.domains.get_all ()):
			p += deterministic_cbn.find_prob (Event ({A: a}), Event ({UA: ua})) * \
				 deterministic_cbn.find_prob (Event ({M: m}), Event ({S: s, A: a, UM: um})) * \
				 deterministic_cbn.find_prob (Event ({N: n}), Event ({S: s, A: a, UN: un})) * \
				 deterministic_cbn.find_prob (Event ({Y: y}), Event ({S: s, A: a, N: n, M: m, UY: uy})) * \
				 ps
	logging.info ("Pearl's three steps: %f" % p)
	return p


def compute_from_observed(s, sprime, y, aprime, mprime):
	probabilistic_cbn = load_xml_to_cbn (cwd + '/../data/synthetic/ProbabilisticBayesianModel.xml')
	probabilistic_cbn.build_joint_table ()

	A = probabilistic_cbn.v['A']
	S = probabilistic_cbn.v['S']
	N = probabilistic_cbn.v['N']
	M = probabilistic_cbn.v['M']
	Y = probabilistic_cbn.v['Y']

	# Let's compute a counterfactual statement that is identifiable
	# print ('-' * 20)

	if s == sprime:
		probabilistic_cbn.build_joint_table ()
		# print ('Identifiable:')
		# print ('Compute according the Bayesian network: '),
		# print (probabilistic_cbn.get_conditional_prob (Event ({Y: y}), Event ({A: aprime, M: mprime, S: sprime})))
		return probabilistic_cbn.get_conditional_prob (Event ({Y: y}), Event ({A: aprime, M: mprime, S: sprime}))
	else:
		# print ('Unidentifiable:')
		p_u = 0.0
		p_l = 0.0
		for n in N.domains.get_all ():
			p_max = -1
			p_min = 2
			for m in M.domains.get_all ():
				p_m = probabilistic_cbn.find_prob (Event ({Y: y}), Event ({A: aprime, N: n, M: m, S: s}))
				p_max = max (p_m, p_max)
				p_min = min (p_m, p_min)
			# print(p_max, p_min)
			p_n = probabilistic_cbn.find_prob (Event ({N: n}), Event ({A: aprime, S: s}))
			p_u += p_max * p_n
			p_l += p_min * p_n

		logging.info ('Upper bound of counterfactual: %f' % p_u)
		logging.info ('Lower bound of counterfactual: %f' % p_l)
		return p_u, p_l


def detect_after_remove(cbn, s, sprime, y, aprime, mprime):
	A = cbn.v['A']
	S = cbn.v['S']
	N = cbn.v['N']
	M = cbn.v['M']
	Y = cbn.v['Y']

	if s == sprime:
		cbn.build_joint_table ()
		logging.info ('Identifiable:')
		logging.info ('Compute according the Bayesian network: '),
		p = cbn.get_conditional_prob (Event ({Y: y}), Event ({A: aprime, M: mprime, S: sprime}))
		logging.info (p)
		return p
	else:
		logging.info ('Unidentifiable:')
		p_u = 0.0
		p_l = 0.0
		for n in N.domains.get_all ():
			p_max = -1
			p_min = 2
			for m in M.domains.get_all ():
				p_m = cbn.find_prob (Event ({Y: y}), Event ({A: aprime, N: n, M: m, S: s}))
				p_max = max (p_m, p_max)
				p_min = min (p_m, p_min)
			# print(p_max, p_min)
			p_n = cbn.find_prob (Event ({N: n}), Event ({A: aprime, S: s}))
			p_u += p_max * p_n
			p_l += p_min * p_n

		logging.info ('Upper bound of counterfactual: %f' % p_u)
		logging.info ('Lower bound of counterfactual: %f' % p_l)
		return p_u, p_l


def pearl_after_remove(data, cbn, s, sprime, y, aprime, mprime):
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
	p = 0.0

	for ua, un, um, us in product (UA.domains.get_all (), UN.domains.get_all (), UM.domains.get_all (), US.domains.get_all ()):
		# compute p(u|z, s)
		ps = data.get_conditional_prob (
			Event ({'UA': ua.index, 'UN': un.index, 'UM': um.index, 'US': us.index}),
			Event ({'A': aprime, 'M': mprime, 'S': sprime}))

		for a, n, m in product (A.domains.get_all (), N.domains.get_all (), M.domains.get_all ()):
			p += cbn.find_prob (Event ({A: a}), Event ({UA: ua})) * \
				 cbn.find_prob (Event ({M: m}), Event ({S: s, A: a, UM: um})) * \
				 cbn.find_prob (Event ({N: n}), Event ({S: s, A: a, UN: un})) * \
				 cbn.find_prob (Event ({Y: y}), Event ({S: s, A: a, N: n, M: m, UY: 1})) * \
				 ps
	logging.info ("Pearl's three steps: %f" % p)
	return p


if __name__ == '__main__':
	tau = 0.05
	spos = 1
	sneg = 0

	# s- -> s+
	ce1 = pd.DataFrame (data=np.zeros ((4, 3)), columns=['ub', 'lb', 'truth'])
	for i, (aprime, mprime) in enumerate (product ([0, 1], [0, 1])):
		p_u, p_l = compute_from_observed (s=spos, sprime=sneg, y=1, aprime=aprime, mprime=mprime)
		p = compute_from_observed (s=sneg, sprime=sneg, y=1, aprime=aprime, mprime=mprime)
		ce1.iloc[i] = (p_u - p), (p_l - p), (pearl_three_step (s=spos, sprime=sneg, y=1, aprime=aprime, mprime=mprime) - pearl_three_step (s=sneg, sprime=sneg, y=1, aprime=aprime, mprime=mprime))

	# s- -> s-
	ce2 = pd.DataFrame (data=np.zeros ((4, 3)), columns=['ub', 'lb', 'truth'])
	for i, (aprime, mprime) in enumerate (product ([0, 1], [0, 1])):
		p_u, p_l = compute_from_observed (s=sneg, sprime=spos, y=1, aprime=aprime, mprime=mprime)
		p = compute_from_observed (s=spos, sprime=spos, y=1, aprime=aprime, mprime=mprime)
		ce2.iloc[i] = (p_u - p), (p_l - p), (pearl_three_step (s=sneg, sprime=spos, y=1, aprime=aprime, mprime=mprime) - pearl_three_step (s=spos, sprime=spos, y=1, aprime=aprime, mprime=mprime))

	tab = pd.concat ([ce1, ce2], axis=1)
	tab = tab.ix[:, 0:3]
	tab.to_csv ('output/synthetic_detect_result.csv', index=False)
	print (tab.round (decimals=3))
