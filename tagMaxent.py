import time
import re
from collections import defaultdict
import argparse
import codecs
from nltk import TaggerI, FreqDist, untag
from nltk.classify.maxent import MaxentClassifier
                  
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dir", required=True,
	help="path to the folder data, example: python tagMaxent.py --dir C:\Users\ tamhv\Desktop\ ")


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



class MaxentPosTagger(TaggerI):
    
    def train(self, train_sents, algorithm='GIS', rare_word_cutoff=5,
              rare_feat_cutoff=5, uppercase_letters='[A-Z]', trace=3,
              **cutoffs):

        self.uppercase_letters = uppercase_letters
        self.word_freqdist = self.gen_word_freqs(train_sents)
        self.featuresets = self.gen_featsets(train_sents,
                rare_word_cutoff)
        self.features_freqdist = self.gen_feat_freqs(self.featuresets)
        self.cutoff_rare_feats(self.featuresets, rare_feat_cutoff)

        t1 = time.time()
        self.classifier = MaxentClassifier.train(self.featuresets, algorithm,
                                                 trace, **cutoffs)
        t2 = time.time()
        if trace > 0:
            print "time to train the classifier: {0}".format(round(t2-t1, 3))

    def gen_feat_freqs(self, featuresets):
        
        features_freqdist = defaultdict(int)
        for (feat_dict, tag) in featuresets:
            for (feature, value) in feat_dict.items():
                features_freqdist[ ((feature, value), tag) ] += 1
        return features_freqdist

    def gen_word_freqs(self, train_sents):
        
        word_freqdist = FreqDist()
        for tagged_sent in train_sents:
            for (word, _tag) in tagged_sent:
                word_freqdist[word] += 1
        return word_freqdist

    def gen_featsets(self, train_sents, rare_word_cutoff):
        
        featuresets = []
        for tagged_sent in train_sents:
            history = []
            untagged_sent = untag(tagged_sent)
            for (i, (_word, tag)) in enumerate(tagged_sent):
                featuresets.append( (self.extract_feats(untagged_sent, i,
                    history, rare_word_cutoff), tag) )
                history.append(tag)
        return featuresets


    def cutoff_rare_feats(self, featuresets, rare_feat_cutoff):
        
        never_cutoff_features = set(['w','t'])

        for (feat_dict, tag) in featuresets:
            for (feature, value) in feat_dict.items():
                feat_value_tag = ((feature, value), tag)
                if self.features_freqdist[feat_value_tag] < rare_feat_cutoff:
                    if feature not in never_cutoff_features:
                        feat_dict.pop(feature)


    def extract_feats(self, sentence, i, history, rare_word_cutoff=5):
        
        features = {}
        hyphen = re.compile("-")
        number = re.compile("\d")
        uppercase = re.compile(self.uppercase_letters)

        #get features: w-1, w-2, t-1, t-2.
        #takes care of the beginning of a sentence
        if i == 0: #first word of sentence
            features.update({"w-1": "<START>", "t-1": "<START>",
                             "w-2": "<START>", "t-2 t-1": "<START> <START>"})
        elif i == 1: #second word of sentence
            features.update({"w-1": sentence[i-1], "t-1": history[i-1],
                             "w-2": "<START>",
                             "t-2 t-1": "<START> %s" % (history[i-1])})
        else:
            features.update({"w-1": sentence[i-1], "t-1": history[i-1],
                "w-2": sentence[i-2],
                "t-2 t-1": "%s %s" % (history[i-2], history[i-1])})

        #get features: w+1, w+2. takes care of the end of a sentence.
        for inc in [1, 2]:
            try:
                features["w+%i" % (inc)] = sentence[i+inc]
            except IndexError:
                features["w+%i" % (inc)] = "<END>"

        if self.word_freqdist[sentence[i]] >= rare_word_cutoff:
            #additional features for 'non-rare' words
            features["w"] = sentence[i]

        else: #additional features for 'rare' or 'unseen' words
            features.update({"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:], "suffix(3)": sentence[i][-3:],
                "suffix(4)": sentence[i][-4:], "prefix(1)": sentence[i][:1],
                "prefix(2)": sentence[i][:2], "prefix(3)": sentence[i][:3],
                "prefix(4)": sentence[i][:4]})
            if hyphen.search(sentence[i]) != None:
                #set True, if regex is found at least once
                features["contains-hyphen"] = True
            if number.search(sentence[i]) != None:
                features["contains-number"] = True
            if uppercase.search(sentence[i]) != None:
                features["contains-uppercase"] = True

        return features


    def tag(self, sentence, rare_word_cutoff=5):
        
        history = []
        for i in xrange(len(sentence)):
            featureset = self.extract_feats(sentence, i, history,
                                               rare_word_cutoff)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


def demo(corpus, num_sents):
    
    if corpus.lower() == "brown":
        from nltk.corpus import brown
        tagged_sents = brown.tagged_sents()[:num_sents]
    elif corpus.lower() == "treebank":
        from nltk.corpus import treebank
        tagged_sents = treebank.tagged_sents()[:num_sents]
    else:
        print "Please load either the 'brown' or the 'treebank' corpus."

    size = int(len(tagged_sents) * 0.1)
    train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
    maxent_tagger = MaxentPosTagger()
    maxent_tagger.train(train_sents)
    
    print "tagger accuracy (test %i sentences, after training %i):" % \
        (size, (num_sents - size)), maxent_tagger.evaluate(test_sents)
    print "\n\n"
    print "classify unseen sentence: ", maxent_tagger.tag(["This", "is", "so",
        "slow", "!"])
    print "\n\n"
    print "show the 10 most informative features:"
    print maxent_tagger.classifier.show_most_informative_features(10)


if __name__ == '__main__':
    #demo("treebank", 200)
    #~ featuresets = demo_debugger("treebank", 10000)
    #print "\n\n\n"
    maxent_tagger = MaxentPosTagger()
    maxent_tagger.train(dataTrain)
    
    try:
        f=open(dir+"\\resultMaxent.txt","w")	
    except IOError:
        print "khong tim thay file"
    f.writelines('[\n')
    for sent in dataTest:
        resultTags=maxent_tagger.tag(sent)
        f.writelines(str(resultTags).encode('utf-8')+',\n')
	
    f.writelines(']')
    f.close()
    print 'Da hoan thanh, hay kiem tra noi dung tag trong file resultMaxent.txt'
    
    


