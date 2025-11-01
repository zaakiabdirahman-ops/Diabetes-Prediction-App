import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

diabete=pd.read_csv("diabetes_prediction_dataset.csv")

diabete.head()

diabete.shape

diabete.columns

diabete.info()

diabete.describe()

diabete.isnull().sum()

diabete.duplicated().sum()

diabete['smoking_history'].value_counts()

def add_counts(ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

# Set up the matplotlib figure
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Plot gender grouped by diabetes
ax = sns.countplot(ax=axes[0, 0], x='gender', hue='diabetes', data=diabete)
axes[0, 0].set_title('Gender grouped by Diabetes')
add_counts(ax)

# Plot hypertension grouped by diabetes
ax = sns.countplot(ax=axes[0, 1], x='hypertension', hue='diabetes', data=diabete)
axes[0, 1].set_title('Hypertension grouped by Diabetes')
add_counts(ax)

# Plot heart disease grouped by diabetes
ax = sns.countplot(ax=axes[1, 0], x='heart_disease', hue='diabetes', data=diabete)
axes[1, 0].set_title('Heart Disease grouped by Diabetes')
add_counts(ax)

# Plot smoking history grouped by diabetes
ax = sns.countplot(ax=axes[1, 1], x='smoking_history', hue='diabetes', data=diabete)
axes[1, 1].set_title('Smoking History grouped by Diabetes')
add_counts(ax)

# Plot diabetes
ax = sns.countplot(ax=axes[2, 0], x='diabetes', data=diabete)
axes[2, 0].set_title('Diabetes Count')
add_counts(ax)

# Create pie plot for diabetes
diabetes_counts = diabete['diabetes'].value_counts()
axes[2, 1].pie(diabetes_counts, labels=diabetes_counts.index, autopct='%1.1f%%', startangle=90)
axes[2, 1].set_title('Diabetes Distribution')
axes[2, 1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
axes[2, 1].legend(title='Diabetes:', loc='upper right')
# Adjust the layout
plt.tight_layout()

# Show the plots
plt.show()

cross_table = pd.crosstab(diabete['diabetes'], diabete['smoking_history'])

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# Plotting the cross table as a heatmap
sns.heatmap(cross_table, cmap='YlOrRd', annot=True, fmt='d', linewidths=0.5, linecolor='black', ax=ax[0])
ax[0].set_title('Diabetes and Smoking History (Heatmap)')
ax[0].set_xlabel('Smoking History')
ax[0].set_ylabel('Diabetes')

# Plotting the cross table with separate bars for smoking history
cross_table.plot(kind='bar', stacked=False, ax=ax[1], color=plt.cm.Paired.colors)
ax[1].set_title('Diabetes and Smoking History (Bar Plot)')
ax[1].set_xlabel('Diabetes')
ax[1].set_ylabel('Count')
ax[1].legend(title='Smoking History', bbox_to_anchor=(1.05, 1), loc='upper left')

# Annotate bars with their values
for container in ax[1].containers:
    ax[1].bar_label(container)

plt.tight_layout()
plt.show()

import sklearn
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in diabete.columns:
  if diabete[i].dtype=='object':
    diabete[i]=le.fit_transform(diabete[i])

diabete.head()

diabete['smoking_history'].value_counts()

corr_matrix=diabete.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f",linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

diabete['bmi'].max()

diabete['bmi'].min()

diabete['HbA1c_level'].max()

diabete['HbA1c_level'].min()

diabete.drop(['gender','smoking_history'],axis=1,inplace=True)

X=diabete.drop('diabetes',axis=1)
Y=diabete['diabetes']

diabete.columns

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
models=[
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    GradientBoostingClassifier(),
    KNeighborsClassifier()
]
for model in models:
  model.fit(x_train,y_train)
  y_pred=model.predict(x_test)
  acc=accuracy_score(y_test,y_pred)
  print(f"Model: {model.__class__.__name__}")
  print(f"ACC: {acc}")
  print()

from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
y_pred_gbc=gbc.predict(x_test)

from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, y_pred_gbc)
recall = recall_score(y_test, y_pred_gbc)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred_gbc==y_test).sum()/len(y_pred_gbc), 3)))

# prompt: report classifier

from sklearn.metrics import classification_report

# Predict the labels for the test set
y_pred = gbc.predict(x_test)

# Print the classification report
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)

from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred_rf==y_test).sum()/len(y_pred_rf), 3)))

Output=pd.DataFrame({'Actual':y_test,'Predicted':y_pred_gbc})
Output

wrong_pred=Output[Output['Actual']!=Output['Predicted']]
len(wrong_pred)

prediction=gbc.predict([[1,1,0,35.5,7.2,210]])
print(prediction)

diabete.head()
diabete['diabetes'].value_counts()

prediction=gbc.predict([[22,0,0,26.8,7.0,85]])
print(prediction)

diabete.columns

diabete.head()
pickle.dump(gbc,open("diabetes_model.pkl","wb"))