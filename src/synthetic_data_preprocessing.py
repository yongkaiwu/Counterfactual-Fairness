import os
import random
import re
import xml.etree.ElementTree as XMLParser

import numpy as np
import pandas as pd

from model.loadxml import load_xml_to_cbn

cwd = os.getcwd ()


def get_deterministic_model():
	# read the original model and transfer it to a deterministic model with all the hidden variable
	# shutil.copyfile (cwd + '/../data/synthetic/OriginalGraph.xml', cwd + '/../data/synthetic/DeterministicGraph.xml')

	fin = open (cwd + '/../data/synthetic/OriginalBayesianModel.xml', 'r')
	fout = open (cwd + '/../data/synthetic/DeterministicBayesianModel.xml', 'w')

	random.seed (12345)
	flag = False
	for line in fin.readlines ():
		if line.find ('<cpt') != -1:
			flag = True
		if line.find ('</cpt') != -1:
			flag = False
		if line.find ('<cpt variable="U') != -1:
			flag = False

		if line.find ('<row>') == -1 or flag == False:
			fout.write (line)
		else:
			start = re.findall (r'^\s*<row>', line)
			new_line = start[0]
			integers = re.findall (r'\d+\.\d+', line)
			new_integers = [0.0] * integers.__len__ ()
			choice = random.randint (0, integers.__len__ () - 1)
			new_integers[choice] = 1.0
			new_line += ' '.join (map (str, new_integers))
			end = re.findall (r'</row>$', line)
			new_line += end[0] + '\n'
			fout.write (new_line)


def get_probabilistic_mode():
	# # delete all hidden nodes from the deterministic graph
	# fin = open (cwd + '/../data/synthetic/DeterministicGraph.xml', 'r')
	# fout = open (cwd + '/../data/synthetic/ProbabilisticGraph.xml', 'w')
	# for line in fin.readlines ():
	# 	if line.find ('>U') == -1:
	# 		fout.write (line)
	# fin.close()
	# fout.close()

	# read the deterministic model and transfer it to a probabilistic model without all the hidden variable
	deterministic_cbn = load_xml_to_cbn (cwd + '/../data/synthetic/DeterministicBayesianModel.xml')

	bayesian_net = XMLParser.parse (cwd + '/../data/synthetic/DeterministicBayesianModel.xml')
	root = bayesian_net.getroot ()

	# vars
	for v in list (root[0])[::- 1]:
		if v.attrib['name'].startswith ('U'):
			root[0].remove (v)

	# causal graph
	for pf in list (root[1])[::- 1]:
		if pf.attrib['name'].startswith ('U'):
			root[1].remove (pf)
		else:
			for p in list (pf)[::-1]:
				if p.attrib['name'].startswith ('U'):
					pf.remove (p)

	# cpts
	for c in root[2][::- 1]:
		name = c.attrib['variable']

		if name.startswith ('U'):
			root[2].remove (c)
		else:
			index = deterministic_cbn.v[name].index
			u_index = list (deterministic_cbn.index_graph.pred[index].keys ())[-1]

			m, n = deterministic_cbn.cpts[index].shape
			u_num = deterministic_cbn.v[u_index].domain_size

			del c[m // u_num:]
			c.attrib['numRows'] = str (m // u_num)

			cpt = pd.DataFrame (data=np.zeros ((m // u_num, n)), columns=deterministic_cbn.cpts[index].columns)
			for i in range (m // u_num):
				cpt.iloc[i] = np.dot (deterministic_cbn.cpts[u_index], deterministic_cbn.cpts[index].iloc[i * u_num:(i + 1) * u_num, :])
				c[i].text = ' '.join (map ("{:.3f}".format, cpt.iloc[i]))

	bayesian_net.write (cwd + '/../data/synthetic/ProbabilisticBayesianModel.xml')


# Generate data by sampling using Tetrad.


def preprocessing():
	# split the data into a training dataset and a testing dataset
	df = pd.read_csv (os.path.join (cwd + '/../data/synthetic/synthetic_data.txt'), sep='\t')

	# split for 80% and 20%
	np.random.seed (2018102411)
	msk = np.random.rand (df.__len__ ()) < 0.8
	train = df[msk]
	test = df[~msk]

	train.to_csv (os.path.join (cwd + '/../data/synthetic/synthetic_train.txt'), sep='\t', index=False)
	test.to_csv (os.path.join (cwd + '/../data/synthetic/synthetic_test.txt'), sep='\t', index=False)


if __name__ == '__main__':
	get_deterministic_model ()
	get_probabilistic_mode ()
	preprocessing ()
