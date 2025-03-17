import numpy as np
import pandas as pd

def create_dataframe():
    df = pd.read_csv('Report_Card_Assessment_Data_2023-24_School_Year_20241104.csv', low_memory=False)
    df_WA = df.loc[df.DistrictName=='State Total']
    df = df.loc[df.DistrictName=='Seattle School District No. 1']
    df_SEA = df.loc[df.SchoolName=='District Total']
    df = df.loc[df.SchoolName.str.contains('|'.join(['Elementary', 'International School']))]
    df['SchoolName'] = df['SchoolName'].str.replace(" International School", ""). \
                                        str.replace(" Elementary School", "").\
                                        str.replace(" Elementary", "")
    df = pd.concat([df, df_SEA, df_WA])
    df = df.loc[df.StudentGroup.isin(['All Students', \
        'Low-Income', 'Non-Low Income', \
        'Hispanic/ Latino of any race(s)', 'White', \
        'English Language Learners', 'Non-English Language Learners'])]
    df = df.loc[(df.GradeLevel=='All Grades') & \
                (df.TestAdministration=='SBAC')]
    df = df[['SchoolName', \
             'StudentGroup', \
             'GradeLevel', \
             'TestAdministration', \
             'TestSubject', \
             'Percent Consistent Grade Level Knowledge And Above']]
    return df

def create_income(df, subject):
    df = df.loc[df.StudentGroup.isin(['Low-Income', \
                                      'Non-Low Income', \
                                      'All Students'])]
    df = df.loc[df.TestSubject==subject]
    df = df[['SchoolName', \
             'StudentGroup', \
             'Percent Consistent Grade Level Knowledge And Above']]
    df = df.pivot(index='SchoolName', \
                  columns='StudentGroup', \
                  values='Percent Consistent Grade Level Knowledge And Above'). \
         reset_index()
    df.dropna(inplace=True)
    df = df.loc[(df['Low-Income'] != 'N<10')]
    df['Low-Income'] = [s.split('%')[0].split('>')[-1].split('<')[-1] \
        for s in df['Low-Income']]
    df['Non-Low Income'] = [s.split('%')[0].split('>')[-1].split('<')[-1] \
        for s in df['Non-Low Income']]
    df['All Students'] = [s.split('%')[0].split('>')[-1].split('<')[-1] \
        for s in df['All Students']]
    df['Low-Income'] = df['Low-Income'].astype(float)
    df['Non-Low Income'] = df['Non-Low Income'].astype(float)
    df['All Students'] = df['All Students'].astype(float)
    df['gap'] = df['Non-Low Income'] - df['Low-Income']
    return df

def create_race(df, subject):
    df = df.loc[df.StudentGroup.isin(['Hispanic/ Latino of any race(s)', \
                                      'White', \
                                      'All Students'])]
    df = df.loc[df.TestSubject==subject]
    df = df[['SchoolName', \
             'StudentGroup', \
             'Percent Consistent Grade Level Knowledge And Above']]
    df= df.pivot(index='SchoolName', \
                 columns='StudentGroup', \
                 values='Percent Consistent Grade Level Knowledge And Above'). \
        reset_index()
    df.dropna(inplace=True)
    df = df.loc[(df['Hispanic/ Latino of any race(s)'] != 'N<10')]
    df['Hispanic/ Latino of any race(s)'] = [s.split('%')[0].split('>')[-1].split('<')[-1] \
        for s in  df['Hispanic/ Latino of any race(s)']]
    df['White'] = [s.split('%')[0].split('>')[-1].split('<')[-1] \
        for s in df['White']]
    df['All Students'] = [s.split('%')[0].split('>')[-1].split('<')[-1] \
        for s in df['All Students']]
    df['Hispanic/ Latino of any race(s)'] = df['Hispanic/ Latino of any race(s)'].astype(float)
    df['White'] = df['White'].astype(float)
    df['All Students'] = df['All Students'].astype(float)
    df['gap'] = df['White'] - df['Hispanic/ Latino of any race(s)']
    return df

def create_ell(df, subject):
    df = df.loc[df.StudentGroup.isin(['English Language Learners', \
                                      'Non-English Language Learners', \
                                      'All Students'])]
    df = df.loc[df.TestSubject==subject]
    df = df[['SchoolName', \
             'StudentGroup', \
             'Percent Consistent Grade Level Knowledge And Above']]
    df = df.pivot(index='SchoolName', \
                  columns='StudentGroup', \
                  values='Percent Consistent Grade Level Knowledge And Above'). \
        reset_index()
    df.dropna(inplace=True)
    df = df.loc[(df['English Language Learners'] != 'N<10')]
    df['English Language Learners'] = [s.split('%')[0].split('>')[-1].split('<')[-1] \
        for s in  df['English Language Learners']]
    df['Non-English Language Learners'] = [s.split('%')[0].split('>')[-1].split('<')[-1] \
        for s in df['Non-English Language Learners']]
    df['All Students'] = [s.split('%')[0].split('>')[-1].split('<')[-1] \
        for s in df['All Students']]
    df['English Language Learners'] = df['English Language Learners'].astype(float)
    df['Non-English Language Learners'] = df['Non-English Language Learners'].astype(float)
    df['All Students'] = df['All Students'].astype(float)
    df['gap'] = df['Non-English Language Learners'] - df['English Language Learners']
    return df
