import jsonlines

G_features={'Nocalls':NoCallsValue,'NoInstruc':NoInstrucValue,'NoStoreData':NoStoreDataValue,'NoLogicalComp':NoLogicalCompValue,'NoTransFun':NoTransFunValue,'NoComp':NoCompValue,'NoArimFun':NoArimFunValue}
G_opt={'H':1,'L':0}
G_compiler={'icc':0,'clang':1,'gcc':2}

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