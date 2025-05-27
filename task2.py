import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv(r'C:\Users\kiran\Downloads\archive\Titanic-Dataset.csv')

print("Basic Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include='all'))

print("\nMissing Values:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include='number').columns

df[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.show()

cols = ['Age', 'Fare', 'Pclass', 'Survived']
existing_cols = [col for col in cols if col in df.columns]

if len(existing_cols) >= 2:
    sns.pairplot(df[existing_cols].dropna(), hue='Survived' if 'Survived' in df.columns else None)
    plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

if 'Pclass' in df.columns and 'Survived' in df.columns:
    sns.barplot(data=df, x='Pclass', y='Survived')
    plt.title("Survival Rate by Passenger Class")
    plt.show()

if 'Age' in df.columns and 'Survived' in df.columns:
    sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
    plt.title("Age Distribution by Survival")
    plt.show()

if 'Sex' in df.columns and 'Survived' in df.columns:
    sns.countplot(data=df, x='Sex', hue='Survived')
    plt.title("Survival Count by Sex")
    plt.show()

if 'Sex' in df.columns and 'Age' in df.columns and 'Survived' in df.columns:
    fig = px.box(df, x='Sex', y='Age', color='Survived', title="Age vs Sex Colored by Survival")
    fig.show()
