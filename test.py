import pandas as pd


from sklearn import tree

train=pd.read_csv("csvjson.csv",header=1)



##features windspeed

##labels pm10

features=[]

labels=[]




for  i in train.values:
    #print(i)

    features.append([float(i[6]),float(i[7])])
    labels.append(float(i[1]))

#print(len(features))
#print(len(labels))
#print(features)



clf=tree.DecisionTreeRegressor()

clf=clf.fit(features,labels)




print(clf.predict([[4.25,1015.2]]))


