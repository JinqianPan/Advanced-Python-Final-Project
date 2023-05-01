#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Usage:
    $ spark-submit --deploy-mode client data_preprocessing_pyspark.py

'''

from pyspark.sql import SparkSession
import time

def load_data(spark, years: list):

    column_names = ['year', 'pistol', 'riflshot', 'asltweap', 'machgun', 
                    'knifcuti', 'othrweap', 'pct', 'trhsloc', 
                    'ac_assoc', 'ac_cgdir', 'ac_rept', 'ac_evasv', 'ac_incid', 
                    'ac_inves', 'ac_proxm', 'ac_time', 'ac_stsnd', 'ac_other',
                    'cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'cs_cloth', 
                    'cs_drgtr', 'cs_furtv', 'cs_vcrim', 'cs_bulge', 'cs_other', 
                    'age', 'build', 'sex', 'ht_feet', 'ht_inch', 'weight', 
                    'inout', 'radio', 'perobs', 'datestop', 'timestop']
    
    for year in years:
        filename = f'./data/{year}.csv'
        print(filename)
        if year == years[0]:
            sqf_data = spark.read.csv(filename, header=True, nullValue=' ')
            sqf_data = sqf_data.select(column_names)
            print( "shape: ", sqf_data.count())
        else:
            this_data = spark.read.csv(filename, header=True, nullValue=' ')
            this_data = this_data.select(column_names)
            sqf_data.union(this_data)
            print( "shape: ", sqf_data.count())

    return sqf_data

def main(spark, years: list):
    start_time = time.time()

    sqf_data = load_data(spark, years)
    sqf_data.show()
    print( "shape: ", sqf_data.count())

    print("--- Took: %s seconds ---\n" % (time.time()-start_time))

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('preprocessing').getOrCreate()

    # years = [2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016]
    years = [2008, 2009]
    main(spark, years)