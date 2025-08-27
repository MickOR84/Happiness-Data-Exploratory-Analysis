# Import tools ('pip install ___', one by one in the terminal prior to running script)
# pip install = 'scikit-learn', 'seaborn', 'pandas', 'matplotlib', 'numpy'

from scipy.stats import f_oneway
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV (local file)
df = pd.read_excel(
    'Happiness_Project\Survey_2_15_16_17.xlsx', engine='openpyxl')

# Inspect first three rows to check correct format
print(df.head(3))
print("\nShape(rows, columns):", df.shape)

# Check data types correspond with changes made in excel, prior to python import
print("\nData Types:")
print(df.dtypes)
print("\nColumn Names:")
print(list(df.columns))

# View summary descriptive statistics
print("\nDescriptives:")
print(df.describe())
print("\nSummary Info:")
(df.info())

# Change data type from integer to string (as year is a label)
df['Year'] = df['Year'].astype(str)

# create object with survey year 2015 only, creating a filtered dataframe containing 2015 data
df_2015 = df[df['Year'] == '2015']
print("2015 data")
# Inspect new dataframe
print(df_2015.head(10))

# Inspect total counts of country within regions
region_counts = df_2015.groupby('Region')['Country'].nunique()
region_include = ['Australia and New Zealand', 'Central and Eastern Europe', 'Eastern Asia', 'Latin America and Caribbean',
                  'Middle East and Northern Africa', 'North America', 'Southeastern Asia', 'Southern Asia', 'Sub-Saharan Africa', 'Western Europe']
region_counts = region_counts.loc[region_include]
print(f"{region_include}:{region_counts}")

# Calculate global happiness score mean
global_mean_15 = df_2015['Happiness Score'].mean()
print(global_mean_15)

# calculate mean average happiness score for western european region only
western_europe = df_2015[df_2015['Region'] == 'Western Europe']
western_europe_mean = western_europe['Happiness Score'].mean()

# Filter and display western european countries with a mean score below the overall western european average mean
below_avg_western_europe = western_europe[western_europe['Happiness Score']
                                          < western_europe_mean]
print("Below Average Western European Countries:")
print(below_avg_western_europe)

# Create scatterplot of below average wesrtern european countries
below_avg_western_europe.plot(x='Country', y='Economy (GDP per Capita)', kind='scatter',
                              figsize=(10, 6),
                              title="Scatter Plot")

plt.show()

# filter and display above average western european mean happiness countries
above_avg_western_europe = western_europe[western_europe['Happiness Score']
                                          > western_europe_mean]
print("Above Average Western European Countries:")
print(above_avg_western_europe)

# create scatterplot for above average western european countries
above_avg_western_europe.plot(x='Country', y='Economy (GDP per Capita)', kind='scatter',
                              figsize=(10, 6),
                              title="Scatterplot")
plt.xticks(rotation=90)
plt.show()


# create pairwise comparisons to understand relationships between happiness indicators
# pair column chart and scatterplot to inspect relationships.
# **General trends for all indicate a positive linear relationship, though with large spreads
cols = ['Economy (GDP per Capita)', 'Health (Life Expectancy)',
        'Freedom', 'Happiness Score']

sns.pairplot(df_2015[cols], kind="reg")
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

# Inspect correlation coefficients to determine effect sizes influencing relationship predicictors
# Freedom low correlation with all variables (happiness score: .57, life expectancy: .36, GDP per capita: .37)
# GDP per capita high correllation with happiness score (.82), and life expectancy (.82)
# Life expectancy as above, though including a medium (high end of medium) correllation with happiness score (.72)
# Higher freedom scores do not predict higher happiness scores (weak positive correlation).
# GDP per capita, life expectancy do predict higher happiness scores (strong positive correllation)
corr_matrix = df_2015[cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Create linear regression to understand how actualscores and predicted scores correlate
# This checks the predictive reliability of the model, to add validity to cormatrix statistics
# Features and target
X = df_2015[['Economy (GDP per Capita)',
             'Health (Life Expectancy)', 'Freedom']]
y = df_2015['Happiness Score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"R2 score: {r2_score(y_test, y_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")

# Coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coef_df)
# Plot axis and define visual output
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Actual vs Predicted Happiness Score')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
corr_coef = np.corrcoef(y_test, y_pred)[0, 1]

# Add correlation coefficient to visual
# Coefficient between actual and predicted scores is high (.87).
# This indicates the predictive model has 87% accuracy for measuring relationships beetween happiness predictors

plt.text(
    0.05, 0.95,
    f"Correlation (r): {corr_coef:.3f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top'
)
plt.show()

# Example: Extract happiness scores by Region groups into separate arrays
groups = [group["Happiness Score"].values for name,
          group in df_2015.groupby("Region")]

# Calculate predictors for all regions
f_stat, p_val = f_oneway(*groups)
print(f"F-statistic: {f_stat}, p-value: {p_val}")

# Results show different predictors for happiness scores in 2015 globally
# Indicating Western Europe happiness predictors are not representative of all countries
# Freedom is the strongest predictor coefficient of happiness globally (2.31).
# The model explains 73% of the variance (r2 = .728)
# Remember that it was the lowest of the predictors used in the western europe regression model (.38)
# Western Europe ranks 2nd in the 2025 Freedom Index (https://r.search.yahoo.com/_ylt=AwrIeazf1K5oUAIA5wcM34lQ;_ylu=Y29sbwNpcjIEcG9zAzEEdnRpZAMEc2VjA3Nj/RV=2/RE=1757497824/RO=10/RU=https%3a%2f%2fworldpopulationreview.com%2fcountry-rankings%2ffreedom-index-by-country%23%3a~%3atext%3dThe%2520regions%2520with%2520the%2520highest%2520levels%2520of%2520freedom%2cand%2520North%2520Africa%252C%2520sub-Saharan%2520Africa%252C%2520and%2520South%2520Asia./RK=2/RS=IkFi98KKmla2WqLX4MECxO4xr5k-)
# This ranks Western Europe only behind North America.
# From this analysis we could hypothesise the higher the individual has freedom, the less likely acknowledgement of freedoms contributions to happiness are subjectively evident.

# Further post-hoc analysis is required before such conclusions can be reached however
