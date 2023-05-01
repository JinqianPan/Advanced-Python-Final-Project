#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Usage:
    $ spark-submit --deploy-mode client data_preprocessing_pyspark.py

'''

from pyspark.sql import SparkSession
import time

def load_data(spark, years: list):

    # schema = 'year STRING, pistol STRING, riflshot STRING, asltweap STRING, \
    #     machgun STRING, knifcuti STRING, othrweap STRING, pct STRING, \
    #     trhsloc STRING, ac_assoc STRING, ac_cgdir STRING, ac_rept STRING, \
    #     ac_evasv STRING, ac_incid STRING, ac_inves STRING, ac_proxm STRING, \
    #     ac_time STRING, ac_stsnd STRING, ac_other STRING, cs_objcs STRING, \
    #     cs_descr STRING, cs_casng STRING, cs_lkout STRING, cs_cloth STRING, \
    #     cs_drgtr STRING, cs_furtv STRING, cs_vcrim STRING, cs_bulge STRING, \
    #     cs_other STRING, age STRING, build STRING, sex STRING, ht_feet STRING, \
    #     ht_inch STRING, weight STRING, inout STRING, radio STRING, \
    #     perobs STRING, datestop STRING, timestop STRING'
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
        if year == years[0]:
            dataframes = spark.read.csv(filename, header=True, nullValue=' ')
        else:
            this_data = spark.read.csv(filename, header=True, nullValue=' ')
            dataframes.union(this_data)
    sqf_data = dataframes

    sqf_data.select(column_names)

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