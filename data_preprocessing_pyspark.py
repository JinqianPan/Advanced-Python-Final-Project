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
    
    def prepro(this_data):
        if year in range(2011, 2017):
                this_data = this_data.drop('forceuse')
        if year in range(2013, 2017):
            this_data = this_data.withColumnRenamed('dettypCM', 'dettypcm')\
                                    .withColumnRenamed('lineCM', 'linecm')\
                                    .withColumnRenamed('detailCM', 'detailcm')
        return this_data

    for year in years:
        filename = f'./data/{year}.csv'
        if year == years[0]:
            sqf_data = spark.read.csv(filename, header=True, nullValue=' ')
            sqf_data = prepro(sqf_data)
        else:
            this_data = spark.read.csv(filename, header=True, nullValue=' ')
            this_data = prepro(this_data)
            sqf_data.union(this_data)

    sqf_data = sqf_data.select(column_names)

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

    years = [2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016]

    main(spark, years)