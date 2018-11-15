import copy

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




Class vertibi_trellis():
    def __init__(self, Ptt, Pwt, test_words, tag_dict):
        self.trellis = []
        self.Ptt = Ptt
        self.Pwt = Pwt
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
    trellis = vertibi_trellis(Ptt, Pwt, test_words, tag_dict)
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
	Read_train(train_file)
	Smoother()
	Read_test(test_file)
	Viterbi()
	Posterior()

if __name__=="__main__":
	main(sys.argv[1], sys.argv[2])
