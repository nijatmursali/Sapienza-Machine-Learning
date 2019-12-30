import jsonlines
import io
import numpy as np
from klepto.archives import file_archive
import random
import sklearn.metrics 
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

strinngfeatures=['NoInstruc','Nocalls','NoStoreData','NoLogicalComp','NoTransFun','NoComp','NoArimFun']
 
NoCallsValue=0
NoInstrucValue=0
NoStoreDataValue=0
NoLogicalCompValue=0
NoTransFunValue=0
NoCompValue=0
NoArimFunValue=0

G_features={'Nocalls':NoCallsValue,'NoInstruc':NoInstrucValue,'NoStoreData':NoStoreDataValue,'NoLogicalComp':NoLogicalCompValue,'NoTransFun':NoTransFunValue,'NoComp':NoCompValue,'NoArimFun':NoArimFunValue}
G_opt={'H':1,'L':0}
G_compiler={'icc':0,'clang':1,'gcc':2}
def Nocalls():
    G_features['NoStoreData']+=1
def NoStoreData():
    G_features['Nocalls']+=1
def NoLogicalComp():
    G_features['NoLogicalComp']+=1
def NoTransFun():
    G_features['NoTransFun']+=1
def NoInstruc(l_instructions):
    G_features['NoInstruc']=len(l_instructions)
def NoComp():
    G_features['NoComp']+=1
def NoArimFun():
    G_features['NoArimFun']+=1

l_feature={'call':Nocalls,'push':NoStoreData,   'test':NoLogicalComp,   'je':NoTransFun     ,'cmovae':NoComp    ,'sub':NoArimFun
                         ,'mov':NoStoreData ,   'cmp':NoLogicalComp ,   'jmp':NoTransFun                        ,'add':NoArimFun
                         ,'pop':NoStoreData ,   'xor':NoLogicalComp ,   'jne':NoTransFun                        ,'imul':NoArimFun
                         ,'shl':NoStoreData                         ,   'ret':NoTransFun
                         ,'lea':NoStoreData                         ,   'seta':NoTransFun
                         ,'movsxd':NoStoreData                      ,   'jg':NoTransFun
                         ,'movzx':NoStoreData  
                         ,'movsxd':NoStoreData  
          } 
nodefined=[]
def specif_features(l_instructions):
    #for xinstruction in l_instructions:
    if l_instructions[0] in l_feature:
        l_value=l_feature[l_instructions[0]]()
    else:
        if l_instructions[0] in nodefined:
            pass
        else:
            #nodefined.append(l_instructions[0])
            #print(str(len(nodefined))+" instruction didn't find  "+l_instructions[0])
            pass
        
    


def Features(instructions):
    l_instructions=[]
    #Dataset2=dict()#{'instructions' : []}
    l_features=dict()

    for instruction in instructions['instructions']:
        splitinstr=instruction.split(' ')
        l_instructions.append(splitinstr)
        specif_features(splitinstr)

    NoInstruc(l_instructions)
    f1=[]
    for xsf in strinngfeatures:
        f1.append(G_features[xsf])
        G_features[xsf]=0
    f2=np.array(f1)
    instructions['instructions']=l_instructions    
    f3.append(f2)
    instructions['features']=l_features
    return instructions
p_DataSet=[]
c1=0
f3=[]
y1=[]
y2=[]
with jsonlines.open("train_dataset.jsonl") as reader:
    for _instructions in reader:#(type=dict, skip_invalid=True):
        p_DataSet.append(Features(_instructions))
        y1.append(G_opt[_instructions['opt']])
        y2.append(G_compiler[_instructions['compiler']])
#np.array([int(c) for c in ])

db = file_archive("DbMalware.txt")
db['featuresnames']=strinngfeatures
db['featuresdata']=f3
db['opt']=y1
db['G_opt']=G_opt
db['compiler']=y2
db['G_compiler']=G_compiler
db.dump()
print('done')

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

print("Libraries imported.")

db = file_archive("DbMalware.txt")
db.load()

strinngfeatures=db['featuresnames']
f3=db['featuresdata']
opt=db['opt']        
G_opt=db['G_opt']      
compiler=db['compiler']  
G_compiler=db['G_compiler'] 
db.drop()

print('done')

DB = datasets.load_iris()
dataset_name = "Iris"
DB = datasets.load_digits()
dataset_name = "Digits"

class_names = np.array(strinngfeatures)#[str(c) for c in DB.target_names])
nodata=30000
kj=DB.data
lp=DB.target
X_all = np.array(f3[0:nodata])#DB.data
y_all = np.array(opt[0:nodata])#DB.target

print(X_all.shape)
print(y_all.shape)

print("Dataset: %s" %(dataset_name))
print("Number of attributes/features: %d" %(X_all.shape[1]))
print("Number of classes: %d %s" %(len(class_names), str(class_names)))
print("Number of samples: %d" %(X_all.shape[0]))

id = random.randrange(0,X_all.shape[0])

print("x%d = %r" %(id,X_all[id]))
print("y%d = %r ['%s']" %(id,y_all[id],class_names[y_all[id]]))


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.333, 
                                                    random_state=14)

print("Size of training set: %d" %X_train.shape[0])
print("Size of test set: %d" %X_test.shape[0])

model = svm.SVC(kernel='linear', probability=True, tol=0.001, C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = model.score(X_test, y_test)
precisSVM = metrics.precision_score(y_test, y_pred)
recallSVM = metrics.recall_score(y_test, y_pred)

print("Accuracy for Support Vector Machines: %.3f" %acc)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision: ", precisSVM)
print("Precision: ", recallSVM)

""" Accuracy for Decision Trees """
decTreeClass = DecisionTreeClassifier()
decTreeClass = decTreeClass.fit(X_train, y_train)

y_predDecisionTree = decTreeClass.predict(X_test)
print("Accuracy for Decision Tree: ", metrics.accuracy_score(y_test, y_predDecisionTree))

pass


