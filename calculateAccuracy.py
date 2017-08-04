import nltk
import sys
import ast
import argparse
import codecs


parser = argparse.ArgumentParser(description="example: python calculateAccuracy.py  result.txt testData2.txt")    
parser.add_argument('file', type=argparse.FileType('r'), nargs='+')

args = parser.parse_args()

if len(args.file)==2:

    try:
        result = args.file[0].read().replace('\t','').decode('utf-8')
        result=eval(result)
        args.file[0].close()
    except IOError:
        print "khong tim thay file"


    try:
        dataTest=args.file[1].read().replace('\t','').decode('utf-8')
        dataTest=eval(dataTest)
        args.file[1].close()
    except IOError:
        print "khong tim thay file"
else:
    print "loi duong dan file  \nexample: python calculateAccuracy.py  result.txt testData.txt"
    exit()

	

def checkSent(sentA, sentB):
    countTrue=0
    if len(sentA) == len(sentB):
        for i in range(len(sentA)):
            if sentA[i][0] != sentB[i][0]:
                return False
            else :
                if sentA[i][1]==sentB[i][1]:
                    countTrue+=1
        return len(sentA),countTrue
    else:
        return False

def calculateAccuracy (result,dataTest):

    if len(result)>0 and len (dataTest)>0 :
        countTotal=0
        countTrue =0
        for sentA in result:
            for sentB in dataTest:
                if(checkSent(sentA,sentB)==False):
                    continue
                else:
                    a,b = checkSent(sentA,sentB)
                    countTotal +=a
                    countTrue +=b                    
                    break  
        return float(countTrue)/countTotal   
    else:
        print "file du lieu khong du de tinh do chinh xac"

print("do chinh xac la: ",calculateAccuracy(result,dataTest))
