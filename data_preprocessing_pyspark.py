#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Usage:
    $ spark-submit --deploy-mode client data_preprocessing_pyspark.py

'''

from pyspark.sql import SparkSession
import time

def load_data(spark, years: list):

    schema = 'year INT, pistol STRING, riflshot STRING, asltweap STRING, \
        machgun STRING, knifcuti STRING, othrweap STRING, pct INT, \
        trhsloc STRING, ac_assoc STRING, ac_cgdir STRING, ac_rept STRING, \
        ac_evasv STRING, ac_incid STRING, ac_inves STRING, ac_proxm STRING, \
        ac_time STRING, ac_stsnd STRING, ac_other STRING, cs_objcs STRING, \
        cs_descr STRING, cs_casng STRING, cs_lkout STRING, cs_cloth STRING, \
        cs_drgtr STRING, cs_furtv STRING, cs_vcrim STRING, cs_bulge STRING, \
        cs_other STRING, age INT, build STRING, sex STRING, ht_feet INT, \
        ht_inch INT, weight INT, inout STRING, radio STRING, perobs STRING, \
        datestop STRING, timestop STRING'

    dataframes = []
    for year in years:
        filename = f'./data/{year}.csv'
        if year == years[0]:
            dataframes = spark.read.csv(filename, header=True, schema=schema)
        this_data = spark.read.csv(filename, header=True, schema=schema)
        dataframes.union(this_data)
    sqf_data = dataframes[0]

    return sqf_data

def main(spark, years: list):
    start_time = time.time()

    sqf_data = load_data(spark, years)
    print( "shape: ", sqf_data.count())
    sqf_data.show()
    print( "shape: ", sqf_data.count())

    print("--- Took: %s seconds ---\n" % (time.time()-start_time))

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('preprocessing').getOrCreate()

    years = [2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016]

    main(spark, years)