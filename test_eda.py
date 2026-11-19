"""
Munira Abdi
CSE 163 AC
This program tests the data cleaning and exploratory data
analysis functions used in the final project.
"""

import pandas as pd
import math

from data_cleaning import clean_diabetes, clean_income
from data_cleaning import clean_healthcare, merge_datasets
from eda import calculate_correlation, calculate_spearman
from eda import identify_outliers, sensitivity_analysis
from eda import identify_low_access_counties


def test_clean_diabetes() -> None:
    """
    Tests that the cleaned diabetes dataset contains all
    39 Washington counties and the expected columns.
    """
    data = clean_diabetes('Data/diabetes.csv')

    assert data.shape == (39, 3)
    assert list(data.columns) == [
        'County', 'FIPS', 'Diabetes_Prevalence'
    ]
    assert data['County'].nunique() == 39
    assert data['Diabetes_Prevalence'].isna().sum() == 0


def test_clean_income() -> None:
    """
    Tests that the cleaned income dataset contains all
    39 Washington counties and valid income values.
    """
    data = clean_income('Data/income.csv')

    assert data.shape == (39, 2)
    assert list(data.columns) == [
        'County', 'Median_Household_Income'
    ]
    assert data['County'].nunique() == 39
    assert data['Median_Household_Income'].isna().sum() == 0
    assert (data['Median_Household_Income'] > 0).all()


def test_clean_healthcare() -> None:
    """
    Tests that the cleaned healthcare dataset contains all
    expected columns and no missing county names.
    """
    data = clean_healthcare('Data/healthcare.xlsx')

    assert list(data.columns) == [
        'County', 'FIPS', 'Physician_Rate'
    ]
    assert data['County'].isna().sum() == 0


def test_merge_datasets() -> None:
    """
    Tests that merging preserves counties from the diabetes
    dataset and represents missing healthcare values as NaN.
    """
    diabetes = pd.DataFrame({
        'County': ['Adams', 'King'],
        'FIPS': [53001, 53033],
        'Diabetes_Prevalence': [12.0, 7.0]
    })

    income = pd.DataFrame({
        'County': ['Adams', 'King'],
        'Median_Household_Income': [65000, 120000]
    })

    healthcare = pd.DataFrame({
        'County': ['Adams'],
        'FIPS': [53001],
        'Physician_Rate': [34.0]
    })

    merged = merge_datasets(diabetes, income, healthcare)

    assert merged.shape == (2, 5)
    assert merged['Physician_Rate'].isna().sum() == 1
    assert merged.loc[0, 'Physician_Rate'] == 34.0
    assert pd.isna(merged.loc[1, 'Physician_Rate'])


def test_calculate_correlation() -> None:
    """
    Tests that Pearson correlation correctly identifies
    a perfect negative linear relationship and excludes
    missing values.
    """
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, None],
        'y': [8, 6, 4, 2, 1]
    })

    r, p, n = calculate_correlation(data, 'x', 'y')

    assert math.isclose(r, -1.0)
    assert p < 0.05
    assert n == 4


def test_calculate_spearman() -> None:
    """
    Tests that Spearman correlation correctly identifies
    a perfect positive rank relationship and excludes
    missing values.
    """
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, None],
        'y': [2, 4, 8, 16, 32]
    })

    r, p, n = calculate_spearman(data, 'x', 'y')

    assert math.isclose(r, 1.0)
    assert p < 0.05
    assert n == 4


def test_identify_outliers() -> None:
    """
    Tests that the IQR method correctly identifies
    an unusually high value as an outlier.
    """
    data = pd.DataFrame({
        'Values': [10, 11, 12, 13, 14, 100]
    })

    outliers = identify_outliers(data, 'Values')

    assert len(outliers) == 1
    assert outliers['Values'].iloc[0] == 100


def test_sensitivity_analysis() -> None:
    """
    Tests that sensitivity analysis removes potential
    outliers and recalculates the correlation correctly.
    """
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 100],
        'y': [2, 4, 6, 8, 10, 1]
    })

    r, p, n = sensitivity_analysis(data, 'x', 'y')

    assert math.isclose(r, 1.0)
    assert p < 0.05
    assert n == 5


def test_identify_low_access_counties() -> None:
    """
    Tests that counties are identified only when both
    income and physician rates are below their medians.
    Also checks that missing values are excluded.
    """
    data = pd.DataFrame({
        'County': ['A', 'B', 'C', 'D', 'E'],
        'Median_Household_Income': [
            40000, 50000, 70000, 90000, 30000
        ],
        'Physician_Rate': [20, 80, 30, 100, None]
    })

    result = identify_low_access_counties(data)

    assert len(result) == 1
    assert result['County'].iloc[0] == 'A'


def main() -> None:
    """
    Runs the testing functions and displays a message
    if all tests pass.
    """
    test_clean_diabetes()
    test_clean_income()
    test_clean_healthcare()
    test_merge_datasets()

    test_calculate_correlation()
    test_calculate_spearman()
    test_identify_outliers()
    test_sensitivity_analysis()
    test_identify_low_access_counties()

    print('All tests passed!')


if __name__ == '__main__':
    main()
