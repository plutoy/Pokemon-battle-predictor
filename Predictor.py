import numpy as np
import pandas as pd

pokemon_df = pd.read_csv("pokemon.csv")
combats_df = pd.read_csv("combats.csv")
tests_df = pd.read_csv("tests.csv")
pokemon_df.drop(["Name","Generation","Legendary"],axis=1,inplace = True)

#Change the type of pokemon to numerical value
tp=[]
for i in range(len(pokemon_df)):
    tp.append(pokemon_df["Type 1"][i])
tp2=list(set(tp))


for i in range(len(pokemon_df)):
    for j in range(len(tp2)):
        if pokemon_df["Type 1"][i]==tp2[j]:
            pokemon_df["Type 1"][i]=(j+1)
        if pokemon_df["Type 2"][i]==tp2[j]:
            pokemon_df["Type 2"][i]=(j+1)
            
            
            
#Delete the null data
pokemon_df["Type 2"]=pokemon_df["Type 2"].fillna(0)
print(pokemon_df)

#Merge combats table and pokemon table
merged1 = combats_df.merge(pokemon_df, how = "left", left_on = "First_pokemon", right_on = "#")
merged2 = combats_df.merge(pokemon_df, how = "left", left_on = "Second_pokemon", right_on = "#")
for i in merged2.columns : 
    merged2.rename(columns = {i : i + "_2"}, inplace = True)

data_train = pd.concat([merged1,merged2], axis = 1)

data_train.sample(5)

for col in data_train.columns :
    if (data_train[col].dtype == "object") :
        data_train[col] = data_train[col].factorize()[0]  # Encode categorical variables
    if (data_train[col].dtype == "bool") :
        data_train[col] = data_train[col].astype(int)  # Just change bool to int

data_train.sample(5)

for i in range(len(data_train)):
    if data_train["Winner"][i] == data_train["First_pokemon"][i]:
        data_train["Winner"][i]=0
    else:
        data_train["Winner"][i]=1

data_train.sample(5)

data_train.drop(["First_pokemon","Second_pokemon","Winner_2","#","#_2","Second_pokemon_2","First_pokemon_2"],axis=1,inplace=True)
data_train.sample(5)

y=data_train.Winner
X=data_train.drop(["Winner"],axis=1)

from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf1 = RandomForestClassifier(n_estimators = 50)
clf2 =GaussianNB()
clf3 =tree.DecisionTreeClassifier()

print(np.mean(cross_val_score(clf1, X, y, cv = 5)))
print(np.mean(cross_val_score(clf2, X, y, cv = 5)))
print(np.mean(cross_val_score(clf3, X, y, cv = 5)))
