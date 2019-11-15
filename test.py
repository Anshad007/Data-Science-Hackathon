import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

training_set = pd.read_csv('TRAINING.csv')
training_set.fillna(0, inplace = True)






#id = training_set['id'] 
Area = training_set['Area(total)']
Troom = training_set['Troom']
Nbedrooms = training_set['Nbedrooms']
Nbwashrooms = training_set['Nbwashrooms']
Twashrooms = training_set['Twashrooms']
RoofArea = training_set['Roof(Area)']
Lawn = training_set['Lawn(Area)']
Nfloors = training_set['Nfloors']
Api = training_set['API']
Anb = training_set['ANB']
Grade = training_set['Grade']
prices = training_set['Price']

new_training_set = {}

# roof = training_set['roof']
# r = []

# for i in roof:
#     if i == 0:
#         r.append("NO")
#     else:
#         i = i.upper()
#         r.append(i)
   
        
        

new_prices = []
for i in prices:
    new_prices.append(int(i[0:4]))
    


#new_training_set['id'] = id
new_training_set['Area'] = Area
new_training_set['Troom'] = Troom
new_training_set['Nbwashrooms'] = Nbwashrooms
new_training_set['Nbedrooms'] = Nbedrooms
new_training_set['Twashrooms'] = Twashrooms
new_training_set['Roof'] = RoofArea
new_training_set['Lawn'] = Lawn
new_training_set['Nfloors'] = Nfloors
new_training_set['Api'] = Api
new_training_set['Anb'] = Anb
new_training_set['Price'] = new_prices
new_training_set['Grade'] = Grade
new_training_set = pd.DataFrame(new_training_set)





test_set = pd.read_csv('TEST.csv')
test_set.fillna(0, inplace = True)

#idt = test_set['id'] 
Areat = test_set['Area(total)']
Troomt = test_set['Troom']
Nbedroomst = test_set['Nbedrooms']
Nbwashroomst = test_set['Nbwashrooms']
Twashroomst = test_set['Twashrooms']
Rooft = test_set['Roof(Area)']
Lawnt = test_set['Lawn(Area)']
Nfloorst = test_set['Nfloors']
Apit = test_set['API']
Anbt = test_set['ANB']
#rooft = test_set['roof']

new_test_set = {}


# rt = []

# for i in rooft:
#     if i == 0:
#         rt.append("NO")
#     else:
#         i = i.upper()
#         rt.append(i)

pricest = test_set['Price']
new_prices_test = []
for i in pricest:
    new_prices_test.append(int(i[0:4]))

new_test_set['Area'] = Areat
new_test_set['Troom'] = Troomt
new_test_set['Nbedrooms'] = Nbedroomst
new_test_set['Nbwashrooms'] = Nbwashroomst
new_test_set['Twashrooms'] = Twashroomst
#new_test_set['roof'] = rt
new_test_set['Roof'] = Rooft
new_test_set['Lawn'] = Lawnt
new_test_set['Nfloors'] = Nfloorst
new_test_set['Api'] = Apit
new_test_set['Anb'] = Anbt
new_test_set['Price'] = new_prices_test
new_test_set = pd.DataFrame(new_test_set)

X = new_training_set.drop('Grade',axis=1)
y = new_training_set['Grade']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

rfc = RandomForestClassifier(n_estimators = 150, criterion="entropy")
rfc.fit(X,y)

rfc_predict = rfc.predict(new_test_set)

a={}
id=[]
for i in range (1,3300):
    id.append(i)

a['id']=pd.Series(id)
a['Grade']=pd.Series(rfc_predict)
a=pd.DataFrame(a)
a.to_csv("Output.csv",index=False)


#rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
print("=== Classification Report ===")
#print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
#print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())