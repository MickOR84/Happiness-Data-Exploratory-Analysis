Here is a README.md summary of the Python script formatted for clarity and easy editing locally:

***

# Happiness Score Analysis

This project analyzes global happiness data from 2015, using descriptive statistics, visualizations, correlation analysis, and a linear regression model to understand key predictors of happiness scores.

## Setup

Install required libraries before running the script:

```bash
pip install scikit-learn seaborn pandas matplotlib numpy openpyxl
```

## Data Loading & Preprocessing

- Loads data from an Excel file `Happiness_Project/Survey_2_15_16_17.xlsx`.
- Filters data for the year 2015.
- Converts the `Year` column to a string type for grouping.

## Exploratory Data Analysis

- Displays first few rows, data types, column names, and summary statistics.
- Counts countries per region for selected regions.
- Calculates global mean happiness score and mean happiness for Western Europe.
- Identifies Western European countries with below- and above-average happiness scores.
- Scatterplots of GDP per Capita for below- and above-average Western European countries.
- Pairwise relationships and correlation heatmap for key features: Economy (GDP per Capita), Health (Life Expectancy), Freedom, and Happiness Score.

## Linear Regression Model

- Features used: Economy (GDP per Capita), Health (Life Expectancy), and Freedom.
- Target variable: Happiness Score.
- Data split into train/test sets (random_state=42).
- Linear regression model is trained on the train set.
- Model performance evaluated on test set using R² score and mean squared error.
- Regression coefficients extracted and displayed.
- Scatter plot of actual vs predicted happiness scores with correlation coefficient.

### Model Performance

- R² Score: ~0.728 (The model explains 72.8% of variance in happiness scores).
- Mean Squared Error: ~0.362.
- Coefficients indicate the relative influence of predictors:
  - **Freedom**: ~2.31 (strongest positive impact)
  - **Economy (GDP per Capita)**: ~1.39
  - **Health (Life Expectancy)**: ~1.10

## Additional Statistical Analysis

- One-way ANOVA conducted to test if happiness scores differ significantly across world regions.
- F-statistic and p-value reported, indicating global regional differences in happiness.

## Interpretations and Insights

- **Freedom** is the **strongest** global predictor of happiness, despite being the **lowest predictor** for **Western Europe**.
- **GDP per Capita** and **Health** also positively correlate with happiness globally.
- Western Europe ranks second globally on the Freedom Index in 2025, behind North America.
- The analysis suggests individuals with more freedom may subjectively underappreciate its contribution to happiness.
- **Recommendation** to conduct further post-hoc analyses to explore specific regional differences.

***

### Usage

Run the script in an environment with access to the data file. Visual outputs include scatterplots, pairplots, corellation matrix tables, and prediction accuracy graphs. Modify paths or regions as needed for local data variations.

