import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.python.framework import ops


alg = "MPVNN"

ctype = 'BLCA'

np.random.seed(100000)
tf.random.set_seed(10000000)

basedir = '..\\Data\\'
outbasedir = '..\\Out\\'
survindex = 4 #DSS




class CustomConnected(Dense):

    def __init__(self,units,connections,**kwargs):

        self.connections = connections                        

        super(CustomConnected,self).__init__(units,**kwargs)  


    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output



pgenes = set()
with open(basedir+'pi3k-akt.txt', 'r') as f:
    for line in f:
        lineData = line.strip().split("\t")
        pgenes.add(lineData[0])
        pgenes.add(lineData[1])


        
count = 0
countp = 0
counto = 0
samples = []
fpgenes = []
fogenes = []
with open(basedir+ctype+'_exp', 'r') as f:
    for line in f:
        count += 1
        lineData = line.rstrip().split("\t")
        if count == 1:
            colc = 0
            for val in lineData:
                colc += 1
                if colc == 1:
                    continue
                samples.append(val)
            continue
        if lineData[0] in pgenes: 
            countp += 1
            fpgenes.append(lineData[0])
            Xpt = np.zeros((1,colc-1))
            colc1 = 0
            for val in lineData:
                colc1 += 1
                if colc1 == 1:
                    continue
                Xpt[0,colc1-2] = val
            if countp == 1:
                Xp = Xpt    
            else:
                Xp = np.concatenate((Xp,Xpt), axis=0)
        else:
            counto += 1
            fogenes.append(lineData[0])
            Xot = np.zeros((1,colc-1))
            colc1 = 0
            for val in lineData:
                colc1 += 1
                if colc1 == 1:
                    continue
                Xot[0,colc1-2] = val
            if counto == 1:
                Xo = Xot   
            else:
                Xo = np.concatenate((Xo,Xot), axis=0)
            

fallgenes = fpgenes + fogenes           
X =  np.concatenate((Xp,Xo), axis=0)               
X = np.transpose(X)
nrow = X.shape[0]
ncol = X.shape[1]



count = 0    
durations = []
patients = [] 
with open(basedir+ctype+'_survival.txt', 'r') as f:
    for line in f:
        count += 1
        if count == 1:
            continue
        lineData = line.rstrip().split("\t")
        if len(lineData) >= survindex+2 and len(lineData[survindex]) > 0 and len(lineData[survindex+1]) > 0:
            if lineData[1] not in patients:
                durations.append(np.float(lineData[survindex+1]))
                patients.append(lineData[1])

survmedian = np.median(durations)


count = 0
survcat = {}
patients = []
with open(basedir+ctype+'_survival.txt', 'r') as f:
    for line in f:
        count += 1
        if count == 1:
            continue
        lineData = line.rstrip().split("\t")
        if lineData[0] in samples:
            if lineData[1] in patients:
                continue
            if len(lineData) >= 11 and lineData[10] == "Redacted":
                continue
            if len(lineData) < survindex+2:
                continue
            if lineData[survindex] == "1" and len(lineData[survindex+1]) > 0:
                if int(lineData[survindex+1]) > survmedian:
                    survcat[lineData[0]] = 0
                    patients.append(lineData[1])
                else:
                    survcat[lineData[0]] = 1
                    patients.append(lineData[1])
            elif lineData[survindex] == "0" and len(lineData[survindex+1]) > 0:
                if int(lineData[survindex+1]) > survmedian:
                    survcat[lineData[0]] = 0
                    patients.append(lineData[1])       
            

y = np.zeros((nrow,1))
count =  0
delindex = []        
for sample in samples:
    if sample in survcat:
        y[count,0] = survcat[sample]
    else:
        delindex.append(count)        
    count += 1

  
X = np.delete(X,delindex,0)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
y = np.delete(y,delindex,0)
y = np.ravel(y)


count = 0
mtngenes = []
with open(basedir+ctype+'_mtn', 'r') as f:
    for line in f:
        count += 1
        if count == 1:
            continue
        lineData = line.rstrip().split("\t")
        mtngenes.append(lineData[0])
        mtnt = np.array(lineData[1:]).astype(np.float)
        if count == 2:
            mtn = mtnt[np.newaxis,:]
        else:
            mtn = np.concatenate((mtn,mtnt[np.newaxis,:]),axis=0)

mtn = np.transpose(mtn)
totmtn = np.sum(mtn,axis=0)
nztotmtn = totmtn[totmtn != 0]
thrmtn = np.quantile(nztotmtn,0.99)
topmtngenes = set()
for i in range(len(totmtn)):
    if totmtn[i] >= thrmtn:
        topmtngenes.add(mtngenes[i])


thrp = len(topmtngenes.intersection(set(fpgenes)))/len(topmtngenes)


edgesp = []
with open(basedir+'pi3k-akt.txt', 'r') as f:
    for line in f:
        lineData = line.rstrip().split("\t")
        if lineData[0] not in fpgenes or lineData[1] not in fpgenes:
            continue
        edgesp.append(lineData[0]+"#"+lineData[1]) 


numnodes = ncol
skf = StratifiedKFold(n_splits=5)
aurocm = []
wcount = 0
allweights1 = []
allweights2 = []


for j in range(20):
    W = np.identity(numnodes)
    newedges = []
    for edge in edgesp:
        ab = edge.split("#")
        rnum1 = np.random.rand(1)[0]
        if rnum1 < thrp:
            while True:
                rnum2 = np.random.rand(1)[0]
                if rnum2 >= thrp:
                    ab = np.random.choice(fpgenes,2,replace=False)
                else:
                    ab = np.random.choice(fogenes,2,replace=False)
                newedge = ab[0] + "#" + ab[1]
                newedgerev = ab[1] + "#" + ab[0]
                if newedge not in edgesp and newedge not in newedges and newedgerev not in edgesp and newedgerev not in newedges:
                    newedges.append(newedge)
                    break        
        W[fallgenes.index(ab[0]),fallgenes.index(ab[1])] = 1
        W[fallgenes.index(ab[1]),fallgenes.index(ab[0])] = 1
        
        
    

    for i in range(20):
        auroc = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model = Sequential()
            callback = EarlyStopping(monitor='val_loss',mode='min',patience=20,verbose=0)
            model.add(CustomConnected(numnodes, W, input_dim=numnodes, activation='tanh'))                                  
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(loss='binary_crossentropy', optimizer='sgd')        
            history = model.fit(X_train, y_train, epochs=500, validation_split=0.3, callbacks =[callback], verbose=0)

            y_pred = model.predict(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            auroc.append(metrics.auc(fpr, tpr))

            wcount += 1
            weights = model.get_weights()
            w1 = np.abs(weights[0]*W)
            w2 = np.abs(weights[2])
            if wcount == 1:
                allweights1 = w1
                allweights2 = w2
            else:
                allweights1 += w1
                allweights2 += w2
                
            del model
            K.clear_session()
            ops.reset_default_graph()

        aurocm.append(np.mean(auroc))


print(alg)    
print(ctype)
print(np.mean(aurocm))
print(np.std(aurocm))

allweights1 = allweights1/wcount
allweights2 = allweights2/wcount
outfilename = outbasedir + alg + '_' + ctype + '_absweightsall.tsv' 
with open(outfilename, 'w') as f:
    for i in range(allweights1.shape[0]):
        for j in range(allweights1.shape[1]):
            val = "\t"
            if j == allweights1.shape[1] - 1:
                val = "\n"
            f.write(str(allweights1[i,j])+val)
    for i in range(allweights2.shape[0]):
        f.write(str(allweights2[i,0])+"\n")


