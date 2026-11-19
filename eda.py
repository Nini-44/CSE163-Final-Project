"""
Munira Abdi
CSE 163 AC
This program performs exploratory data analysis on diabetes
prevalence, household income, and healthcare access across
Washington State counties.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, shapiro, spearmanr

from data_cleaning import clean_diabetes, clean_income
from data_cleaning import clean_healthcare, merge_datasets


def analyze_missing_data(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and displays the number and
    percentage of missing values in each column. Also displays
    the counties with missing primary care physician rates.
    """
    missing = data.isna().sum()
    percentage = (missing / len(data)) * 100

    print(f'Missing values:\n{missing}')
    print(f'Percentage of missing values:\n{percentage.round(2)}')

    missing_counties = data[data['Physician_Rate'].isna()]
    print(f'Counties with missing physician rates:\n{missing_counties}')


def summarize_variables(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and displays summary statistics
    for diabetes prevalence, median household income, and
    primary care physician rates. Missing values are ignored
    when calculating the statistics.
    """
    variables = data[['Diabetes_Prevalence',
                      'Median_Household_Income',
                      'Physician_Rate']]

    summary = variables.describe()
    print(f'Summary statistics:\n{summary.round(2)}')


def plot_diabetes_prevalence(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and creates a horizontal bar chart
    showing age-adjusted diabetes prevalence across Washington
    State counties. Sorts the counties by prevalence, displays
    percentage labels, and highlights the highest and lowest rates.
    Saves the chart as a PNG file and returns None.
    """
    sorted_data = data.sort_values('Diabetes_Prevalence')

    plt.figure(figsize=(10, 12))

    bars = plt.barh(sorted_data['County'],
                    sorted_data['Diabetes_Prevalence'],
                    color='steelblue')

    bars[0].set_color('seagreen')
    bars[-1].set_color('firebrick')

    plt.bar_label(bars, fmt='%.1f%%', padding=3)

    plt.title('Adult Diabetes Prevalence Across\n'
              'Washington State Counties (2023)')
    plt.xlabel('Age-Adjusted Diabetes Prevalence (%, 2023)')
    plt.ylabel('County')
    plt.xlim(0, 15)

    plt.legend([bars[-1], bars[0]],
               ['Highest Prevalence', 'Lowest Prevalence'])

    plt.margins(y=0.02)
    plt.savefig('Visualizations/diabetes_prevalence.png')
    plt.close()


def plot_income_diabetes(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and creates a scatter plot showing
    the relationship between median household income and diabetes
    prevalence across Washington State counties. Each point
    represents one county. Saves the plot as a PNG file and returns None.
    """
    plt.figure(figsize=(9, 6))

    plt.scatter(data['Median_Household_Income'],
                data['Diabetes_Prevalence'],
                color='steelblue', edgecolor='black', alpha=0.8)

    plt.title('Median Household Income and Adult Diabetes\n'
              'Prevalence Across Washington State Counties (2023)')
    plt.xlabel('Median Household Income (2023 Dollars)')
    plt.ylabel('Age-Adjusted Diabetes Prevalence (%, 2023)')

    plt.grid(alpha=0.3)
    plt.savefig('Visualizations/income_diabetes.png')
    plt.close()


def plot_healthcare_diabetes(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and creates a scatter plot showing
    the relationship between primary care physician rates and diabetes
    prevalence across Washington State counties. Excludes counties with
    missing values. Each point represents one county.
    Saves the plot as a PNG file and returns None.
    """
    filtered_data = data.dropna(subset=['Physician_Rate',
                                        'Diabetes_Prevalence'])

    plt.figure(figsize=(9, 6))

    plt.scatter(filtered_data['Physician_Rate'],
                filtered_data['Diabetes_Prevalence'],
                color='steelblue', edgecolor='black', alpha=0.8)

    plt.title('Primary Care Physician Availability and Adult\n'
              'Diabetes Prevalence Across Washington State Counties')
    plt.xlabel('Primary Care Physicians per 100,000 Residents (2021)')
    plt.ylabel('Age-Adjusted Diabetes Prevalence (%, 2023)')

    plt.grid(alpha=0.3)
    plt.savefig('Visualizations/healthcare_diabetes.png')
    plt.close()


def plot_income_healthcare(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and creates a scatter plot showing
    the relationship between median household income and primary
    care physician rates across Washington State counties.
    Excludes counties with missing values. Each point represents
    one county. Saves the plot as a PNG file and returns None.
    """
    filtered_data = data.dropna(subset=['Median_Household_Income',
                                        'Physician_Rate'])

    plt.figure(figsize=(9, 6))

    plt.scatter(filtered_data['Median_Household_Income'],
                filtered_data['Physician_Rate'],
                color='steelblue', edgecolor='black', alpha=0.8)

    plt.title('Median Household Income and Primary Care Physician\n'
              'Availability Across Washington State Counties')
    plt.xlabel('Median Household Income (2023 Dollars)')
    plt.ylabel('Primary Care Physicians per 100,000 Residents (2021)')

    plt.grid(alpha=0.3)
    plt.savefig('Visualizations/income_healthcare.png')
    plt.close()


def calculate_correlation(data: pd.DataFrame, x_column: str,
                          y_column: str) -> tuple[float, float, int]:
    """
    Takes a combined DataFrame and two numerical column names.
    Removes rows with missing values and returns the Pearson
    correlation coefficient, p-value, and sample size.
    """
    clean_data = data[[x_column, y_column]].dropna()

    r, p = pearsonr(clean_data[x_column], clean_data[y_column])

    return r, p, len(clean_data)


def calculate_spearman(data: pd.DataFrame, x_column: str,
                       y_column: str) -> tuple[float, float, int]:
    """
    Takes a combined DataFrame and two numerical column names.
    Removes rows with missing values and returns the Spearman
    correlation coefficient, p-value, and sample size.
    """
    clean_data = data[[x_column, y_column]].dropna()

    r, p = spearmanr(clean_data[x_column], clean_data[y_column])

    return r, p, len(clean_data)


def identify_outliers(data: pd.DataFrame,
                      column: str) -> pd.DataFrame:
    """
    Takes a DataFrame and a numerical column name.
    Returns rows containing potential outliers using
    the interquartile range (IQR) method.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[(data[column] < lower_bound) |
                    (data[column] > upper_bound)]

    return outliers


def sensitivity_analysis(data: pd.DataFrame, x_column: str,
                         y_column: str) -> tuple[float, float, int]:
    """
    Takes a combined DataFrame and two numerical column names.
    Removes rows containing potential outliers in either column
    and returns the Pearson correlation coefficient, p-value,
    and sample size after removing the outliers.
    """
    x_outliers = identify_outliers(data, x_column)
    y_outliers = identify_outliers(data, y_column)

    outlier_indices = x_outliers.index.union(y_outliers.index)

    filtered_data = data.drop(outlier_indices)

    return calculate_correlation(filtered_data, x_column, y_column)


def identify_low_access_counties(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a combined DataFrame and returns counties with both
    median household incomes and primary care physician rates
    below their respective medians. Excludes missing values.
    """
    filtered_data = data.dropna(subset=['Median_Household_Income',
                                        'Physician_Rate'])

    median_income = filtered_data['Median_Household_Income'].median()
    median_physicians = filtered_data['Physician_Rate'].median()

    low_access_counties = filtered_data[
        (filtered_data['Median_Household_Income'] < median_income) &
        (filtered_data['Physician_Rate'] < median_physicians)
    ]

    return low_access_counties


def print_correlation_results(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and displays the Pearson
    correlation coefficient, p-value, and sample size
    for research questions 2, 3, and 4.
    """
    comparisons = [
        ('RQ2: Income and Diabetes Prevalence',
         'Median_Household_Income', 'Diabetes_Prevalence'),
        ('RQ3: Physician Availability and Diabetes Prevalence',
         'Physician_Rate', 'Diabetes_Prevalence'),
        ('RQ4: Income and Physician Availability',
         'Median_Household_Income', 'Physician_Rate')
    ]

    for title, x_column, y_column in comparisons:
        correlation, p_value, sample_size = calculate_correlation(
            data, x_column, y_column
        )
        print(title)
        print(f'Pearson r: {correlation:.3f}, '
              f'p-value: {p_value:.4f}, n: {sample_size}')


def print_spearman_results(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and displays the Spearman
    correlation results for research questions 2, 3, and 4.
    """
    comparisons = [
        ('RQ2: Spearman Correlation',
         'Median_Household_Income', 'Diabetes_Prevalence'),
        ('RQ3: Spearman Correlation',
         'Physician_Rate', 'Diabetes_Prevalence'),
        ('RQ4: Spearman Correlation',
         'Median_Household_Income', 'Physician_Rate')
    ]

    for title, x_column, y_column in comparisons:
        correlation, p_value, sample_size = calculate_spearman(
            data, x_column, y_column
        )
        print(title)
        print(f'Spearman r: {correlation:.3f}, '
              f'p-value: {p_value:.4f}, n: {sample_size}')


def print_outlier_results(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and displays potential
    outliers for diabetes prevalence, household income,
    and primary care physician rates.
    """
    print('Diabetes Prevalence Outliers:')
    print(identify_outliers(data, 'Diabetes_Prevalence'))

    print('Median Household Income Outliers:')
    print(identify_outliers(data, 'Median_Household_Income'))

    print('Primary Care Physician Rate Outliers:')
    print(identify_outliers(data, 'Physician_Rate'))


def print_sensitivity_results(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and displays the correlation
    results for research questions 2, 3, and 4 after removing
    potential outliers.
    """
    comparisons = [
        ('RQ2: Sensitivity Analysis',
         'Median_Household_Income', 'Diabetes_Prevalence'),
        ('RQ3: Sensitivity Analysis',
         'Physician_Rate', 'Diabetes_Prevalence'),
        ('RQ4: Sensitivity Analysis',
         'Median_Household_Income', 'Physician_Rate')
    ]

    for title, x_column, y_column in comparisons:
        correlation, p_value, sample_size = sensitivity_analysis(
            data, x_column, y_column
        )
        print(title)
        print(f'Pearson r: {correlation:.3f}, '
              f'p-value: {p_value:.4f}, n: {sample_size}')


def check_normality(data: pd.DataFrame) -> None:
    """
    Takes a combined DataFrame and uses the Shapiro-Wilk test
    to check the normality of diabetes prevalence, median
    household income, and primary care physician rates.
    Excludes missing values and displays the test results.
    """
    columns = ['Diabetes_Prevalence',
               'Median_Household_Income',
               'Physician_Rate']

    for column in columns:
        values = data[column].dropna()
        statistic, p_value = shapiro(values)

        print(f'Normality Test: {column}')
        print(f'Statistic: {statistic:.3f}, '
              f'p-value: {p_value:.4f}')


def main() -> None:
    """
    Loads and merges the cleaned datasets to explore diabetes
    prevalence, household income, and healthcare access.
    """
    diabetes = clean_diabetes('Data/diabetes.csv')
    income = clean_income('Data/income.csv')
    healthcare = clean_healthcare('Data/healthcare.xlsx')
    merged = merge_datasets(diabetes, income, healthcare)

    analyze_missing_data(merged)
    summarize_variables(merged)

    plot_diabetes_prevalence(merged)
    plot_income_diabetes(merged)
    plot_healthcare_diabetes(merged)
    plot_income_healthcare(merged)

    check_normality(merged)
    print_correlation_results(merged)
    print_spearman_results(merged)
    print_outlier_results(merged)
    print_sensitivity_results(merged)

    print('RQ4: Counties with Low Income and Physician Availability')
    print(identify_low_access_counties(merged))


if __name__ == '__main__':
    main()
