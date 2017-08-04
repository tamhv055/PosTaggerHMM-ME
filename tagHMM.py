import nltk
import sys
import ast
import argparse
import codecs



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dir", required=True,
	help="path to the folder data, example: python tagHMM.py --dir C:\Users\ tamhv\Desktop\ ")


dir =ap.parse_args().dir


try:
    f=open(dir+"\\dataTrain.txt","r")
    dataTrain=f.read().replace('\t','').decode('utf-8')
    dataTrain=eval(dataTrain)
    f.close()
except IOError:
    print "khong tim thay file"
	



try:
    f=open(dir+"\\dataTest1.txt","r")
    dataTest=f.read().replace('\t','').decode('utf-8')
    dataTest=eval(dataTest)
    f.close()
except IOError:
    print "khong tim thay file"





#print(dataTrain[0])
#print(dataTrain.decode('utf8'))


tags_words = [ ]
for sent in dataTrain:

	tags_words.append( ("START", "START") )	
	tags_words.extend([ (tag[:3], word) for (word, tag) in sent ])
	tags_words.append( ("END", "END") )


cfd_tagwords = nltk.ConditionalFreqDist(tags_words)
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)
tags = [tag for (tag, word) in tags_words ]

cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(tags))
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
distinct_tags = set(tags)

try:
	f=open(dir+"\\resultHMM.txt","w")
	#f = codecs.open(dir+"\\result.txt", "w", "utf-8")
except IOError:
	print "khong tim thay file"

f.writelines('[\n')
for sent in dataTest:
	
	viterbi = [ ]
	backpointer = [ ]
	first_viterbi = { }
	first_backpointer = { }

	for tag in distinct_tags:
		if tag == "START": continue
		first_viterbi[ tag ] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob( sent[0] )
		first_backpointer[ tag ] = "START"

	viterbi.append(first_viterbi)
	backpointer.append(first_backpointer)
	currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[ tag ])

	for wordindex in range(1, len(sent)):
		this_viterbi = { }
		this_backpointer = { }
		prev_viterbi = viterbi[-1]
		
		for tag in distinct_tags:
			if tag == "START": continue
			best_previous = max(prev_viterbi.keys(),key = lambda prevtag:prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sent[wordindex]))       
			this_viterbi[ tag ] = prev_viterbi[ best_previous] * cpd_tags[ best_previous ].prob(tag) * cpd_tagwords[ tag].prob(sent[wordindex])
			this_backpointer[ tag ] = best_previous

		currbest = max(this_viterbi.keys(), key = lambda tag: this_viterbi[ tag ])
		
		viterbi.append(this_viterbi)
		backpointer.append(this_backpointer)
	

	prev_viterbi = viterbi[-1]
	best_previous = max(prev_viterbi.keys(),key = lambda prevtag: prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob("END"))
	prob_tagsequence = prev_viterbi[ best_previous ] * cpd_tags[ best_previous].prob("END")
	best_tagsequence = [ "END", best_previous ]

	backpointer.reverse()

	current_best_tag = best_previous
	for bp in backpointer:
		best_tagsequence.append(bp[current_best_tag])
		current_best_tag = bp[current_best_tag]

	best_tagsequence.reverse()	
	best_tagsequence= str(best_tagsequence).replace("'START',",'').replace(", 'END'",'')
	best_tagsequence=eval(best_tagsequence)
	
	resultTags=zip(sent,best_tagsequence)
	
	f.writelines(str(resultTags).encode('utf-8')+',\n')
	
f.writelines(']')
f.close()
print 'Da hoan thanh, hay kiem tra noi dung tag trong file resultHMM.txt'


#
#try:
'''    f=open(dir+"\\dataTest2.txt","r")
except IOError:
    print "khong tim thay file"
	
dataTest2=f.read().replace('\t','').decode('utf-8')
dataTest2=eval(dataTest2)
f.close()
'''


#sentA = [(u'Nam',u'NNP'),(u'học',u'VB'),(u'cách',u'NN'),(u'giới-thiệu',u'VB'),(u'sản-phẩm',u'NN')]
#sentB = [(u'Nam',u'NNP'),(u'học',u'VB'),(u'cách',u'NN'),(u'giới-thiệu',u'VB'),(u'sản-phẩm',u'NN')]
        
#print checkSent(sentA,sentB)   

'''
for sentA in result:
    print type(sentA)
    break
'''






