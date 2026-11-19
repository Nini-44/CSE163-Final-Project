"""
Munira Abdi
CSE 163 AC
This program cleans and merges diabetes, income, and
healthcare data for Washington State counties.
"""

import pandas as pd


def clean_diabetes(filename: str) -> pd.DataFrame:
    """
    Takes a CSV filename containing CDC PLACES diabetes data.
    Filters the data to include Washington counties with age-adjusted
    diabetes prevalence and returns a DataFrame containing county
    names, FIPS codes, and diabetes prevalence rates.
    """
    data = pd.read_csv(filename)

    data = data[(data['StateAbbr'] == 'WA') &
                (data['Year'] == 2023) &
                (data['Measure'] == 'Diagnosed diabetes among adults') &
                (data['Data_Value_Type'] == 'Age-adjusted prevalence')]

    data = data[['LocationName', 'LocationID', 'Data_Value']]

    data = data.rename(columns={
        'LocationName': 'County',
        'LocationID': 'FIPS',
        'Data_Value': 'Diabetes_Prevalence'
    })

    return data


def clean_income(filename: str) -> pd.DataFrame:
    """
    Takes a CSV filename containing ACS median household income data.
    Reorganizes the data so each row represents a county and returns
    a DataFrame containing county names and median household incomes.
    """
    data = pd.read_csv(filename)

    counties = []
    incomes = []

    for column in data.columns:
        if column.endswith('!!Estimate'):
            county = column.replace(' County, Washington!!Estimate', '')
            income = data[column].iloc[0]

            counties.append(county)
            incomes.append(income)

    cleaned = pd.DataFrame({
        'County': counties,
        'Median_Household_Income': incomes
    })

    cleaned['Median_Household_Income'] = pd.to_numeric(
        cleaned['Median_Household_Income'].str.replace(',', ''),
        errors='coerce'
    )

    return cleaned


def clean_healthcare(filename: str) -> pd.DataFrame:
    """
    Takes an Excel filename containing County Health Rankings data.
    Removes the statewide record and returns a DataFrame containing
    county names, FIPS codes, and primary care physician rates per
    100,000 residents.
    """
    data = pd.read_excel(filename, sheet_name='Select Measure Data',
                         header=1)

    data = data[data['County'].notna()]

    data = data[['County', 'FIPS', 'Primary Care Physicians Rate']]

    data = data.rename(columns={
        'Primary Care Physicians Rate': 'Physician_Rate'
    })

    return data


def merge_datasets(diabetes: pd.DataFrame, income: pd.DataFrame,
                   healthcare: pd.DataFrame) -> pd.DataFrame:
    """
    Takes three cleaned DataFrames containing diabetes prevalence,
    median household income, and primary care physician rates.
    Merges the datasets using county names and FIPS codes and returns
    a combined DataFrame that keeps all counties from the diabetes
    dataset, and represents missing values as NaN.
    """
    merged = diabetes.merge(income, on='County', how='left')

    merged = merged.merge(healthcare, on=['County', 'FIPS'],
                          how='left')

    return merged


def main() -> None:
    """
    Loads, cleans, and merges the diabetes, income, and healthcare
    datasets. Displays the combined dataset and its number of
    rows and columns.
    """
    diabetes = clean_diabetes('Data/diabetes.csv')
    income = clean_income('Data/income.csv')
    healthcare = clean_healthcare('Data/healthcare.xlsx')
    merged = merge_datasets(diabetes, income, healthcare)

    print(merged.head())
    print(merged.shape)


if __name__ == '__main__':
    main()
