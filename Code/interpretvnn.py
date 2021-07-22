import numpy as np


alg = "MPVNN"

ctype = 'BRCA'

topmax = 3
thrval = 0.75

basedir = '..\\Data\\'
outbasedir = '..\\Out\\'


print(ctype)
edges = []
pgenes = set()
with open(basedir+'pi3k-akt.txt', 'r') as f:
    for line in f:
        lineData = line.strip().split("\t")
        pgenes.add(lineData[0])
        pgenes.add(lineData[1])
        


        
count = 0
fpgenes = []
fogenes = []
with open(basedir+ctype+'_exp', 'r') as f:
    for line in f:
        count += 1
        lineData = line.rstrip().split("\t")
        if count == 1:
            continue
        if lineData[0] in pgenes: 
            fpgenes.append(lineData[0])
        else:
            fogenes.append(lineData[0])
            

fallgenes = fpgenes + fogenes
numgenes = len(fallgenes)


edgesp = []
with open(basedir+'pi3k-akt.txt', 'r') as f:
    for line in f:
        lineData = line.rstrip().split("\t")
        if lineData[0] not in fpgenes or lineData[1] not in fpgenes:
            continue
        edgesp.append(lineData[0]+"#"+lineData[1]) 

count = 0
with open(outbasedir+alg+'_'+ctype+'_absweightsall.tsv', 'r') as f:
    for line in f:
        count += 1
        lineData = line.rstrip().split("\t")
        if count == 1:
            W1 = np.array(lineData).astype(np.float)[np.newaxis,:]
        elif count <= numgenes:
            W1 = np.concatenate((W1,np.array(lineData).astype(np.float)[np.newaxis,:]), axis=0)
        elif count == numgenes+1:
            W2 = np.array(lineData).astype(np.float)[np.newaxis,:]
        else:
            W2 = np.concatenate((W2,np.array(lineData).astype(np.float)[np.newaxis,:]), axis=0)


sindval = (-W2).argsort(axis=0)
topgenes = []
count = 0
for ind in sindval:
    topgenes.append(fallgenes[ind[0]])
    count += 1
    if count == topmax:
        break


#print(topgenes)


tW1 = W1.flatten()
tW1 = tW1[tW1 != 0]
thrmax1 = np.quantile(tW1,thrval)
tW2 = W2.flatten()
tW2 = tW2[tW2 != 0]
thrmax2 = np.quantile(tW2,thrval)
for index in range(topmax):
    print("\nTop Path "+str(index+1))
    candgene = topgenes[index]
    topsgenes = []
    while(1>0):
        found = False
        valmax = thrmax1
        for edge in edgesp:
            ab = edge.split("#")
            if ab[1] == candgene:
                if W1[fallgenes.index(ab[0]),fallgenes.index(ab[1])] >= valmax:
                    found = True
                    valmax = W1[fallgenes.index(ab[0]),fallgenes.index(ab[1])]
                    ncandgene = ab[0]
        if not found:
            break
        else:
            candgene = ncandgene
            if W2[fallgenes.index(candgene),0] < thrmax2:
                break
            topsgenes.append(candgene)


    candgene = topgenes[index]
    toptgenes = []
    while(1>0):
        found = False
        valmax = thrmax1
        for edge in edgesp:
            ab = edge.split("#")
            if ab[0] == candgene:
                if W1[fallgenes.index(ab[1]),fallgenes.index(ab[0])] >= valmax:
                    found = True
                    valmax = W1[fallgenes.index(ab[1]),fallgenes.index(ab[0])]
                    ncandgene = ab[1]
        if not found:
            break
        else:
            candgene = ncandgene
            if W2[fallgenes.index(candgene),0] < thrmax2:
                break
            toptgenes.append(candgene)


    toppathgenes = list(reversed(topsgenes)) + [topgenes[index]] + toptgenes
    
    count = 0
    for gene in (toppathgenes):
        count += 1
        printgene = gene
        if count == len(topsgenes) + 1:
            printgene = gene+"(top)"
        if count == len(toppathgenes):
            print(printgene,end = '')
        else:
            print(printgene+"->",end = '')
    









