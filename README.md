
# Data-Analysis-Project: Sales Performance Analysis

## Project Overview
This repository showcases a comprehensive sales performance analysis project, leveraging data analysis techniques and visualization to derive actionable insights from sales data across various dimensions like time, geography, and product categories.

### Project Goals
- Uncover sales trends and anomalies.
- Predict future sales using historical data.
- Assess marketing campaign effectiveness.

## Tools & Technologies
- **Python**: Core language for data manipulation and analysis.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: For creating insightful data visualizations.
- **Scikit-learn**: For statistical and machine learning models.
- **Jupyter Notebook**: For interactive data exploration and presentation.

## Repository Structure

- **`data/`**: Contains both raw and processed datasets.
  - `sales_raw.csv`: Original sales data.
  - `sales_cleaned.csv`: Processed sales data after cleaning.
  
- **`scripts/`**: Python scripts for various tasks:
  - `data_preprocessing.py`: Data cleaning and feature engineering.
  - `eda_visualization.py`: Exploratory Data Analysis visuals.
  - `modeling.py`: Sales prediction model.

- **`notebooks/`**: Jupyter notebooks for interactive analysis:
  - `sales_eda.ipynb`: Detailed EDA.
  - `sales_modeling.ipynb`: Model creation and evaluation.

- **`reports/`**: Documentation of findings:
  - `sales_analysis_report.pdf`: Comprehensive report on findings.

## How to Use This Project

1. **Clone the Repository:**

2. **Setup Environment:**

3. **Explore Data:**
- Open `sales_eda.ipynb` for an interactive dive into the data.

4. **Run Analysis:**
- Execute the scripts or notebooks to see the data processing, analysis, and modeling steps.

## Key Findings
- Discovered a significant uptick in sales during Q4, possibly linked to holiday shopping.
- Identified underperforming regions for strategic focus.
- Model predicts a 5% increase in sales with current trends for the next quarter.

## Code Samples

### Example from `data_preprocessing.py`

```python
import pandas as pd

def load_and_clean_data(file_path):
 df = pd.read_csv(file_path)
 df = df.dropna()
 df['sales'] = df['sales'].clip(lower=0, upper=df['sales'].quantile(0.99))
 df['date'] = pd.to_datetime(df['date'])
 df['month'] = df['date'].dt.month
 df['year'] = df['date'].dt.year
import seaborn as sns
import matplotlib.pyplot as plt

def plot_sales_trend(df):
    monthly_sales = df.groupby(['year', 'month'])['sales'].sum().reset_index()
    plt.figure(figsize=(15, 7))
    sns.lineplot(x='month', y='sales', hue='year', data=monthly_sales)
    plt.title('Monthly Sales Trend by Year')
    plt.savefig('reports/Monthly_Sales_Trend.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv('data/sales_cleaned.csv')
    plot_sales_trend(df)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_and_evaluate_model(df):
    X = df[['month', 'year', 'marketing_spend']]
    y = df['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    return model

if __name__ == "__main__":
    df = pd.read_csv('data/sales_cleaned.csv')
    trained_model = train_and_evaluate_model(df)

pandas==1.3.4
matplotlib==3.4.3
seaborn==0.11.1
scikit-learn==0.24.2
jupyter==1.0.0
 return df

MIT License

Copyright (c) 2014-present Sebastian McKenzie and other contributors

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

if __name__ == "__main__":
 cleaned_df = load_and_clean_data('data/sales_raw.csv')
 cleaned_df.to_csv('data/sales_cleaned.csv', index=False)
