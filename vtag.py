

def Read_train(train_file):
'''
Count dictionary: Cwt[('w', 't')], Ctt[('t_i', 't_i-1')], Ct['t']
Singuton dictionary: Singtt, Singtw
Tag dictionary: tag_dict['w']
'''
	Cwt, Ctt, Ct, Singtt, Singtw, tag_dict, tag_2_idx
	f = open(train_file, 'r')
	line  = f.readline()
	for line in f:
		w, t = line.strip().split('/')
		if w not
	return Cwt, Ctt, Ct, Singtt, Singtw, tag_dict, tag_2_idx


def Read_test(test_file):
'''
return two lists of same length: test_words, test_tags
'''
	return test_words, test_tags


def Smoother(Curr_Cwt, Curr_Ctt, Curr_Ct, Singtt, Singtw, mode):
'''
mode=1: no smooth
mode=2: One count smoothing
return probablity dictionary: Ptt[('t_i', 't_i-1')], Ptw[('w', 't')]
'''
	return Ptt, Ptw


def Accuracy(gt_tags, pred_tags):
	print("")


def Viterbi(Ptt, Ptw, tag_dict, test_words, test_tags):
'''
maintain numpy di A[t_i, t], B[t_i, i]. Dimension is #tag_type x sentence_length
'''
	Accuracy(test_tags, pred_tags)

def Posterior(Ptt, Ptw, tag_dict):
'''
maintain numpy array U[t_i, t]. Dimension is #tag_type x sentence_length
maintain array BP[t_i, t]. Dimension is #tag_type x sentence_length
'''
	Accuracy(test_tags, pred_tags)

def main(train_file, test_file):
	Read_train(train_file)
	Smoother()
	Read_test(test_file)
	Viterbi()
	Posterior()

if __name__=="__main__":
	main(sys.argv[1], sys.argv[2])