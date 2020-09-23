#################   DO NOT EDIT THESE IMPORTS #################
import math
import random
import numpy
from collections import *

#################   PASTE PROVIDED CODE HERE AS NEEDED   #################
class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix


#####################  STUDENT CODE BELOW THIS LINE  #####################

def compute_counts(training_data, order):
	"""
	:param training_data: a list of (word, POS) pairs
	:param order: the order of the HMM desired
	:return: a tuple containing the number of tokens, the dictionary containing C(ti, wi),
	a dictionary containing C(ti), a dictionary containing C(ti-1, ti), a dictionary containing C(ti-2, ti-1, ti) if order is 3,
	"""
	# the order of HMM is the problem could not be other than 2 or 3
	if order != 2 and order != 3:
		return None
	num_token = len(training_data)
	# initialize the first 3 dictionaries
	dic_1 = defaultdict(lambda: defaultdict(int))
	dic_2 = defaultdict(int)
	dic_3 = defaultdict(lambda: defaultdict(int))
	# run through all the pairs
	for word, tag in training_data:
		dic_1[tag][word] += 1
		dic_2[tag] += 1
	for idx in range(num_token - 1):
		dic_3[training_data[idx][1]][training_data[idx + 1][1]] += 1
	if order == 2:
		return num_token, dic_1, dic_2, dic_3
	else:
		# compute the 4th dictionary
		dic_4 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		for idx in range(num_token - 2):
			dic_4[training_data[idx][1]][training_data[idx + 1][1]][training_data[idx + 2][1]] += 1
		return num_token, dic_1, dic_2, dic_3, dic_4

def compute_initial_distribution(training_data, order):
	"""
	:param training_data: a list of (word, POS) pairs
	:param order: the order of the desired HMM
	:return: the dictionary containing the initial distribution of the HMM
	"""
	# check if order is 2 or 3
	if order != 2 and order != 3:
		return None
	l = len(training_data)
	if order == 2:
		start = training_data[0][1]
		dict_2 = defaultdict(int)
		dict_2[start] += 1
		# count the number of s period
		s = 1
		for idx in range(l - 1):
			if training_data[idx][1] == '.':
				dict_2[training_data[idx + 1][1]] += 1
				s += 1
		# normalize the probability
		for tag in dict_2.keys():
			dict_2[tag] = dict_2[tag] / float(s)
		return dict_2
	else:
		start_1 = training_data[0][1]
		start_2 = training_data[1][1]
		dict_3 = defaultdict(lambda :defaultdict(int))
		dict_3[start_1][start_2] += 1
		# count the number of periods
		s = 1
		for idx in range(l - 2):
			if training_data[idx][1] == '.':
				dict_3[training_data[idx + 1][1]][training_data[idx + 2][1]] += 1
				s += 1
		# normalize
		for tag_1 in dict_3.keys():
			for tag_2 in dict_3[tag_1].keys():
				dict_3[tag_1][tag_2] = dict_3[tag_1][tag_2] / float(s)
		return dict_3

def compute_emission_probabilities(unique_words, unique_tags, W, C):
	"""
	:param unique_words: a set of words that appear in the training data
	:param unique_tags: a set of tags that appear in the training data
	:param W: the dictionary of C(ti, wi)
	:param C: the dictionary of C(ti)
	:return: the dictionary containing the emission probalilities of HMM
	"""
	dict_1 = defaultdict(lambda : defaultdict(int))
	for tag in unique_tags:
		num_tag = C[tag]
		for word in unique_words:
			if W[tag][word] != 0:
				dict_1[tag][word] = W[tag][word] / float(num_tag)
	return dict_1

def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
	"""
	:param unique_tags: a set of all tags in the data
	:param num_tokens: a set of all the words in the data
	:param C1: the dictionary containing C(ti)
	:param C2: the dictionary containing C(ti-1, ti)
	:param C3: the dictionary containing C(ti-2, ti-1, ti)
	:param order: the order of desired HMM
	:return: the lambdas for computation of transition matrix
	"""
	# check if order is 2 or 3
	if order != 2 and order != 3:
		return None
	if order == 3:
		# set the initial value of lambdas
		lamb = [0.0, 0.0, 0.0]
		# iterate over all ti-2, ti-1. ti
		for t_1 in unique_tags:
			for t_2 in unique_tags:
				for t_3 in unique_tags:
					if C3[t_1][t_2][t_3] > 0:
						a = [0, 0, 0]
						# apply the formula
						a[0] = (C1[t_3] - 1) / float(num_tokens)
						# if denominator is 0, set a to 0
						if C1[t_3] > 1:
							a[1] = (C2[t_2][t_3] - 1) / float(C1[t_3] - 1)
						if C2[t_1][t_2] > 1:
							a[2] = (C3[t_1][t_2][t_3] - 1) / float(C2[t_1][t_2] - 1)
						# find argmax_i(ai)
						i = a.index(max(a))
						lamb[i] += C3[t_1][t_2][t_3]
		sum1 = sum(lamb)
		# normalize
		final_lamb = [float(item) / sum1 for item in lamb]
		return final_lamb
	else:
		lamb = [0.0, 0.0, 0.0]
		# iterate over all ti-1, ti
		for t_1 in unique_tags:
			for t_2 in unique_tags:
				# similar as the order 3 case
				if C2[t_1][t_2] > 0:
					a = [0, 0]
					a[0] = (C1[t_2] - 1) / float(num_tokens)
					if C1[t_1] != 1:
						a[1] = (C2[t_1][t_2] - 1) / float(C1[t_1] - 1)
					i = a.index(max(a))
					lamb[i] += C2[t_1][t_2]
		s = sum(lamb)
		for i in range(3):
			lamb[i] = float(lamb[i]) / s
		return lamb

def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
	"""
	:param training_data: a list of (word, POS) pairs
	:param unique_tags: a set of tags appearing in the training data
	:param unique_words: a set of words appearing in the training data
	:param order: the order of desired HMM
	:param use_smoothing: whether using smoothing or not when computing transition matrix
	:return: the built HMM
	"""
	num_tok, w, c1, c2, c3 = compute_counts(training_data, 3)
	#print compute_counts(training_data, 3)
	ini_dis = compute_initial_distribution(training_data, order)
	emi_mat = compute_emission_probabilities(unique_words, unique_tags, w, c1)
	if use_smoothing == False:
		# set the value of lambdas
		if order == 2:
			lambdas = [0, 1, 0]
		if order == 3:
			lambdas = [0, 0, 1]
	else:
		lambdas = compute_lambdas(unique_tags, num_tok, c1, c2, c3, order)
		#print lambdas
	# compute the transition matrix
	if order == 2:
		tran_mat = defaultdict(lambda: defaultdict(int))
		for tag_1 in unique_tags:
			for tag_2 in unique_tags:
				# if the denominator is 0, set the value to be 0
				if c1[tag_1] != 0:
					tran_mat[tag_1][tag_2] = lambdas[1] * c2[tag_1][tag_2] / float(c1[tag_1]) + lambdas[0] * c1[tag_1] / float(num_tok)
				else:
					tran_mat[tag_1][tag_2] = lambdas[0] * c1[tag_1] / float(num_tok)
	else:
		# similar operations as order of 2
		tran_mat = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		for tag_1 in unique_tags:
			for tag_2 in unique_tags:
				for tag_3 in unique_tags:
					if c2[tag_1][tag_2] != 0 and c1[tag_2] != 0:
						tran_mat[tag_1][tag_2][tag_3] = lambdas[2] * c3[tag_1][tag_2][tag_3] / float(c2[tag_1][tag_2]) +  \
														lambdas[1] * c2[tag_2][tag_3] / float(c1[tag_2]) + lambdas[0] * c1[tag_3] / float(num_tok)
					elif c2[tag_1][tag_2] == 0:
						tran_mat[tag_1][tag_2][tag_3] = lambdas[1] * c2[tag_2][tag_3] / float(c1[tag_2]) + lambdas[0] * \
														c1[tag_3] / float(num_tok)
					elif c1[tag_2] == 0:
						tran_mat[tag_1][tag_2][tag_3] = lambdas[2] * c3[tag_1][tag_2][tag_3] / float(c2[tag_1][tag_2]) +  lambdas[0] * \
														c1[tag_3] / float(num_tok)
					else:
						tran_mat[tag_1][tag_2][tag_3] = lambdas[0] * c1[tag_3] / float(num_tok)
	return HMM(order, ini_dis, emi_mat, tran_mat)

def trigram_viterbi(hmm, sentence):
	"""
	:param hmm: an HMM model
	:param sentence: a sentence observed
	:return: a sequence of tag that could best model the observation
	"""
	# initialize the dp matrix
	viterbi = defaultdict(lambda : defaultdict(lambda: defaultdict(int)))
	backpointer = defaultdict(lambda : defaultdict(lambda: defaultdict(int)))
	# get the set of tags
	unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
	#print unique_tags
	for tag_1 in unique_tags:
		for tag_2 in unique_tags:
			# set the marginal values of viterbi matrix
			if (hmm.initial_distribution[tag_1][tag_2] != 0) and (hmm.emission_matrix[tag_1][sentence[0]] != 0) and (hmm. emission_matrix[tag_2][sentence[1]] != 0):
				viterbi[tag_1][tag_2][1] = math.log(hmm.initial_distribution[tag_1][tag_2]) + math.log(hmm.emission_matrix[tag_1][sentence[0]]) + \
										   math.log(hmm.emission_matrix[tag_2][sentence[1]])
			else:
				viterbi[tag_1][tag_2][1] = -1 * float('inf')

	# Dynamic programming.
	for t in range(2, len(sentence)):
		for s in unique_tags:
			# set the backpointer of no_path
			backpointer["No_Path"][s][t] = "No_Path"
		backpointer["No_Path"]["No_Path"][t] = "No_Path"
		for s_1 in unique_tags:
			for s_2 in unique_tags:
				max_value = -1 * float('inf')
				max_state = None
				for s_prime in unique_tags:
					val1 = viterbi[s_prime][s_1][t - 1]
					val2 = -1 * float('inf')
					if hmm.transition_matrix[s_prime][s_1][s_2] != 0:
						val2 = math.log(hmm.transition_matrix[s_prime][s_1][s_2])
					curr_value = val1 + val2
					# find the max_probability
					if curr_value > max_value:
						max_value = curr_value
						max_state = s_prime
				val3 = -1 * float('inf')
				if hmm.emission_matrix[s_2][sentence[t]] != 0:
					val3 = math.log(hmm.emission_matrix[s_2][sentence[t]])
				viterbi[s_1][s_2][t] = max_value + val3
				if max_state == None:
					backpointer[s_1][s_2][t] = "No_Path"
				else:
					backpointer[s_1][s_2][t] = max_state


	# Termination
	max_value = -1 * float('inf')
	last_state = (None, None)
	final_time = len(sentence) - 1
	for s_1 in unique_tags:
		for s_2 in unique_tags:
			if viterbi[s_1][s_2][final_time] > max_value:
				max_value = viterbi[s_1][s_2][final_time]
				last_state = (s_1, s_2)
	if last_state == (None, None):
		last_state = ("No_Path", "No_Path")

	# Traceback
	tagged_sentence = []
	tagged_sentence.append((sentence[len(sentence) - 1], last_state[1]))
	tagged_sentence.append((sentence[len(sentence) - 2], last_state[0]))
	for i in range(len(sentence) - 3, -1, -1):
		next_tag = (tagged_sentence[-1][1], tagged_sentence[-2][1])
		curr_tag = backpointer[next_tag[0]][next_tag[1]][i + 2]
		tagged_sentence.append((sentence[i], curr_tag))
	tagged_sentence.reverse()
	return tagged_sentence

#Test cases

#training_data1 = [('CS','N'), ('is','V'),('harsh','A'),('.','.'), ('but', 'P'), ('it', 'N'), ('is', 'V'), ('doable', 'A'), ('.', '.'), ('yes', 'P'), ('it', 'N'), ('is', 'V'), ('.', '.')]
#training_data2 = [('hw7','N'), ('is','V'),('difficult','A'),('.','.')]
#print compute_counts(training_data1, 2)
#print compute_counts(training_data2, 2)
#print compute_counts(training_data1, 3)
#print compute_counts(training_data2, 3)
#print compute_initial_distribution(training_data1, 2)
#print compute_initial_distribution(training_data2, 2)
#print compute_initial_distribution(training_data1, 3)
#print compute_initial_distribution(training_data2, 3)
#unique_words1 = ['CS', 'is', 'harsh', '.', 'but', 'doable', 'yes', 'it']
#unique_words2 = ['hw7', 'is', 'difficult', '.']
#unique_tags1 = ['N', 'V', 'A', '.', 'P']
#unique_tags2 = ['N', 'V', 'A', '.', 'P']

#num1 = compute_counts(training_data1, 3)[0]
#w1 = compute_counts(training_data1, 3)[1]
#c11 = compute_counts(training_data1, 3)[2]
#c12 = compute_counts(training_data1, 3)[3]
#c13 = compute_counts(training_data1, 3)[4]

# num2 = compute_counts(training_data2, 3)[0]
# w2 = compute_counts(training_data2, 3)[1]
# c21 = compute_counts(training_data2, 3)[2]
# c22 = compute_counts(training_data2, 3)[3]
# c23 = compute_counts(training_data2, 3)[4]

#print compute_emission_probabilities(unique_words1, unique_tags1, w1, c11)
#print compute_emission_probabilities(unique_words2, unique_tags2, w2, c21)

#print compute_lambdas(unique_tags1, num1, c11, c12, c13, 2)
#print compute_lambdas(unique_tags1, num1, c11, c12, c13, 3)
#print compute_lambdas(unique_tags2, num2, c21, c22, c23, 2)
#print compute_lambdas(unique_tags2, num2, c21, c22, c23, 3)

#hmm1 = build_hmm(training_data1, unique_tags1, unique_words1, 2, False)
#hmm2 = build_hmm(training_data2, unique_tags2, unique_words2, 3, True)


#sentence = ['Yes', 'CS', 'is', 'doable', '.']
#print trigram_viterbi(hmm1, sentence)
#print trigram_viterbi(hmm2, sentence)

#training_data = [('hw7','N'), ('is','V'),('difficult','A'),('.','.'), ('but', 'P'), ('it', 'N'), ('is', 'V'), ('doable', 'A'), ('.', '.'), ('yes', 'P'), ('it', 'N'), ('is', 'V'), ('.', '.')]
#training_data = [('hw7','N'), ('is','V'),('difficult','A'),('.','.')]
#print compute_counts(training_data, 3)
#print compute_initial_distribution(training_data, 3)
#unique_words = ['hw7', 'is', 'difficult', '.', 'but', 'doable', 'yes', 'it']
#unique_tags = ['N', 'V', 'A', '.', 'P']
# unique_words = ['hw7', 'is', 'difficult', '.']
# unique_tags = ['N', 'V', 'A', '.']
# num = compute_counts(training_data, 3)[0]
# w = compute_counts(training_data, 2)[1]
# c1 = compute_counts(training_data, 3)[2]
# c2 = compute_counts(training_data, 3)[3]
# c3 = compute_counts(training_data, 3)[4]
#print compute_emission_probabilities(unique_words, unique_tags, w, c)
#print compute_lambdas(unique_tags, num, c1, c2, c3, 2)
#print build_hmm(training_data, unique_tags, unique_words, 2, False).emission_matrix



#hmm = HMM(3, _trigram_initial_distribution, _trigram_emission_probabilities, _trigram_transition_matrix)

#sentence = ['Heads', 'Heads', 'Tails']
#print trigram_viterbi(hmm, sentence)