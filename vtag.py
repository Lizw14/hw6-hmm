

def count_add(entry, dic):
	if entry not in dic:
		dic[entry] = 1
	else:
		dic[entry] += 1

def Read_train(train_file):
'''
Count dictionary: Cwt[('w', 't')], Ctt[('t_i', 't_i-1')], Ct['t']
Singuton dictionary: Singtt, Singtw
Tag dictionary: tag_dict['w']
'''
	Cwt = {}
	Ctt = {}
	Ct = {}
	Cw = {}
	Singtt = {} 
	Singtw = {} 
	tag_dict = {} 
	tag_set = set()
	word_set = set()

	f = open(train_file, 'r')
	line  = f.readline()
	t_last = '###'
	for line in f:
		w, t = line.strip().split('/')

		# Cwt, Singtw
		if (w,t) not in Cwt:
			Cwt[(w,t)] = 1
			if t not in Singtw:
				Singtw[t] = 1
			else:
				Singtw[t] += 1
		elif Cwt[(w,t)] == 1:
			Cwt[(w,t)] +=1
			Singtw[t] -= 1
		else:
			Cwt[(w,t)] +=1

		# Ctt, Singtt
		if (t,t_last) not in Ctt:
			C[(t,t_last)] = 1
			if t_last not in Singtt:
				Singtt[t_last] = 1
			else:
				Singtt[t_last] += 1
		elif Ctt[(t,t_last)] == 1:
			Ctt[(t,t_last)] +=1
			Singtt[t_last] -= 1
		else:
			Ctt[(t,t_last)] +=1

		# Ct
		if t not in Ct:
			Ct[t] = 1
		else:
			Ct[t] += 1

		# Cw
		if w not in Cw:
			Cw[w] = 1
		else:
			Cw[w] += 1

		# tag_dict
		if w not in tag_dict:
			tag_dict = [t]
		elif t not in tag_dict[w]:
			tag_dict[w].append(t)

		# tag_2_idx
		tag_set.add(t)

		# word_list
		word_set.add(w)

		t_last = t
		
	f.close()

	tag_2_idx = {}
	for idx, t in enumerate(tag_set):
		tag_2_idx[t] = idx
		tag_2_idx[idx] = t

	word_set.remove('###')
	V = len(word_set)

	Cw.pop('###')
	N = 0
	for k in Cw:
		N += Cw[k]

	tag_dict['###'] = list(tag_set)

	return Cwt, Ctt, Ct, Cw, Singtt, Singtw, tag_dict, tag_2_idx, tag_set, word_set, N, V


def Read_test(test_file):
'''
return two lists of same length: test_words, test_tags
'''
	test_words = []
	test_tags = []
	f = open(test_file, 'r')
	line  = f.readline()
	for line in f:
		w, t = line.strip().split('/')
		test_words.append(w)
		test_tags.append(t)
	f.close()
	return test_words, test_tags

def log_(a)
	if a==0:
		return -float('inf')
	else:
		return math.log(a)

def Smoother(Cwt, Ctt, Ct, Cw, Singtt, Singtw, word_set, tag_set, N, V, mode):
'''
mode=1: no smooth
mode=2: One count smoothing
return probablity dictionary: Ptt[('t_i', 't_i-1')], Ptw[('w', 't')]
'''
	Ptt = {}
	Ptw = {}

	if mode==1:
		for t in tag_set:
			for t_last in tag_set:
				k = (t, t_last)
				if (Ctt.get(k,0)>0):
					Ptt[k] = math.log(float(Ctt[k])/Ct[k[1]])
				else:
					Ptt[k] = -float('inf')
		for w in word_set:
			for t in tag_set:
				k = (w, t)
				if (Cwt.get(k, 0)>0):
					ptw[k] = math.log(float(Cwt[k])/Ct[k[1]])
				else:
					ptw[k] = -float('inf')

	elif mode==2:
		for t in tag_set:
			for t_last in tag_set:
				k = (t, t_last)
				lam = 1 + Singtt(t_last)
				Ptt[k] = math.log((Ctt.get(k,0) + lam*Ct.get(t,0)/N)/(Ct.get(t_last,0)+lam))
		for w in word_set:
			for t in tag_set:
				k = (w, t)
				lam = 1 + Singtw(t)
				Ptw[k] = math.log((Cwt.get(k,0) + lam*((Cw.get(w,0)+1)/(N+V)))/(Ct.get(t,0)+lam))
			

	for t in tag_set:
		Ptw[('###', t)] = -float('inf')
	Ptw[('###', '###')] = 0

	return Ptt, Ptw


def Accuracy(gt_tags, pred_tags):
	assert len(gt_tags)==len(pred_tags)
	count = 0
	for idx, gt_t in enumerate(gt_tags):
		if gt_t==pred_tags[idx]:
			count += 1
	acc = float(count)/len(gt_tags)
	return acc


class vertibi_trellis():
    def __init__(self, Ptt, Ptw, test_words, tag_dict):
        self.trellis = []
        self.Ptt = Ptt
        self.Ptw = Ptw
        self.tag_dict = tag_dict
        self.test_words = test_words
        self.trellis_length = len(self.test_words)


    def compute_trellis(self):

        for i in range(self.trellis_length):        

            #store all the possible tags and their u and backpointer at current position 
            state_dict = {}
            for tag in self.tag_dict[self.test_words[i]]:
                state_dict[tag] = [0,None]

            #build trellis    
            #trellis[i][1] = tag_dict 
            #tag_dict[key] = [best mu,backpointer]
            self.trellis.append([self.test_words[i],copy.deepcopy(state_dict)])
            
            for tag in self.tag_dict[self.test_words[i]]:
                #loop over tags in previous position
                temp_best = 0
                for tag_1,_ in self.trellis[i-1][1].items():
                    p = Ptt[(tag, tag_1)] + Ptw[(self.test_words[i], tag)]
                    #mu for tag_1 in previous position
                    mu_1 = self.trellis[i-1][1][tag_1][0]
                    mu = p + mu_1
                    
                    if mu > temp_best:
                        self.trellis[i][1][tag][0] = mu
                        self.trellis[i][1][tag][1] = tag_1
                    

    def return_best_path(self):
        best_path = []
        
        #get the best tag in last position:
        best_value = 0
        best_tag = none
        for tag,value in self.trellis[-1][1].items():
            if value[0] > best_value:
                best_value = value[0]
                best_tag = tag

        #insert the best tag for last position
        best_path.insert(0, best_tag)

        for i in range(self.trellis_length, 0, -1):
            last_tag = self.trellis[i][1][best_path[0]][1]
            best_path.insert(0,last_tag)     

        return best_path

def Viterbi(Ptt, Ptw, tag_dict, test_words, test_tags):
'''
maintain numpy di A[t_i, t], B[t_i, i]. Dimension is #tag_type x sentence_length
'''
    trellis = vertibi_trellis(Ptt, Ptw, test_words, tag_dict)
    trellis.compute_trellis()
    pred_tags = trellis.return_best_path()

    acc = Accuracy(test_tags, pred_tags)

    return acc
    

def Posterior(Ptt, Ptw, tag_dict):
'''
maintain numpy array U[t_i, t]. Dimension is #tag_type x sentence_length
maintain array BP[t_i, t]. Dimension is #tag_type x sentence_length
'''
   
	Accuracy(test_tags, pred_tags)

def main(train_file, test_file):
	Cwt, Ctt, Ct, Cw, Singtt, Singtw, tag_dict, tag_2_idx, tag_set, word_set, N, V = Read_train(train_file)
	Ptt, Ptw = Smoother(Cwt, Ctt, Ct, Cw, Singtt, Singtw, word_set, tag_set, N, V, mode=1)
	test_words, test_tags = Read_test(test_file)
	acc = Viterbi(Ptt, Ptw, tag_dict, test_words, test_tags)
	print(acc)
	#Posterior()

if __name__=="__main__":
	main(sys.argv[1], sys.argv[2])