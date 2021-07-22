import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import svm


alg = "SVM"

ctype = 'BLCA'


basedir = '..\\Data\\'
survindex = 4 #DSS




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


skf = StratifiedKFold(n_splits=5)
aurocm = []
wcount = 0
allweights = []


aurocm = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model  = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    aurocm.append(metrics.auc(fpr, tpr))
    
    del model


print(alg)    
print(ctype)
print(np.mean(aurocm))


