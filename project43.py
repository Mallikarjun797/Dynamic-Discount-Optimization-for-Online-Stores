import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#  Load Data
df = pd.read_csv("dynamicdiscount.csv")
print(df.head())
print(df.info())

#  Data Cleaning
df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

#  EDA

#Histplot 
plt.figure(figsize=(8, 5))
sns.histplot(df['discount_percent'], bins=20, kde=True)
plt.title('Distribution of Discount Percent')
plt.xlabel('Discount Percent')
plt.ylabel('Frequency')
plt.show()

#Scatter plot 
plt.figure(figsize=(8, 5))
sns.scatterplot(x='discount_percent', y='final_price', data=df)
plt.title('Discount Percent vs Final Price')
plt.xlabel('Discount Percent')
plt.ylabel('Final Price')
plt.show()

#  Pairplot
plt.figure(figsize=(8, 5))
sns.pairplot(df, vars=['price', 'competitor_price', 'demand_score', 'discount_percent'], hue='stock')
plt.title('Pairplot of Key Features')
plt.show()

# Correlation Heatmap
numerical_cols = ['price', 'competitor_price', 'demand_score', 'final_price', 'discount_percent', 'user_click_rate', 'conversion_rate']
plt.figure(figsize=(10, 7))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Feature Selection and Model Preparation
features = ['price', 'competitor_price', 'demand_score', 'user_click_rate', 'conversion_rate', 'stock']
target = 'discount_percent'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Model Training
model = LinearRegression()
model.fit(X_train, y_train)

#  Model Evaluation
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Discount Percent")
plt.ylabel("Predicted Discount Percent")
plt.title("Actual vs Predicted Discount Percent")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

#  Personalized Discount 
def recommend_discount(user_features_dict):
    user_df = pd.DataFrame([user_features_dict])
    pred_discount = model.predict(user_df)[0]
    return max(0, round(pred_discount, 2)) 

#  Real-time user/product scenario
new_user = {'price': 600,'competitor_price': 620, 'demand_score': 0.75, 'user_click_rate': 0.15,'conversion_rate': 0.05,'stock': 300}
personalized_discount = recommend_discount(new_user)
print(f"Recommended Personalized Discount: {personalized_discount}%")




