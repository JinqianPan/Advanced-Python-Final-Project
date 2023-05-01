#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Usage:
    $ spark-submit --deploy-mode client data_preprocessing_pyspark.py

'''

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, BooleanType, StringType
from pyspark.sql.functions import when
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
        else:
            this_data = spark.read.csv(filename, header=True, nullValue=' ')
            this_data = this_data.select(column_names)
            sqf_data = sqf_data.union(this_data)

    return sqf_data

def main(spark, years: list):
    start_time = time.time()

    coln = ["cs_objcs", "cs_descr", "cs_casng","cs_lkout", "cs_cloth", 
                "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other",
                "ac_rept", "ac_inves", "ac_proxm", "ac_evasv", "ac_assoc", 
                "ac_cgdir", "ac_incid", "ac_time", "ac_stsnd", "ac_other", 
                "pistol", "riflshot", "asltweap", "knifcuti", "machgun", 
                "othrweap"]
    
    drop_column_names = ['pistol', 'riflshot', 'asltweap', 'machgun', 
                'knifcuti', 'othrweap', 'pct', 'trhsloc', 
                'ac_assoc', 'ac_cgdir', 'ac_rept', 'ac_evasv', 'ac_incid', 
                'ac_inves', 'ac_proxm', 'ac_time', 'ac_stsnd', 'ac_other',
                'cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'cs_cloth', 
                'cs_drgtr', 'cs_furtv', 'cs_vcrim', 'cs_bulge', 'cs_other', 
                'age', 'build', 'sex', 'ht_feet', 'ht_inch', 'weight', 
                'inout', 'radio', 'perobs', 'datestop', 'timestop', 
                'found_pistol', 'found_rifle', 'found_assault', 
                'found_machinegun', 'found_knife', 'found_other']
    
    recode_yn_udf = udf(lambda f: True if f == 'Y' else False if f == 'N' else None, BooleanType())
    recode_io_udf = udf(lambda f: True if f == 'I' else False if f == 'O' else None, BooleanType())


    sqf_data = load_data(spark, years)
    print( "shape: ", sqf_data.count() )

    sqf_data = sqf_data.na.drop(subset=['timestop'])

    sqf_data = sqf_data \
        .withColumn('month', sqf_data['datestop'].substr(1, 2).cast(IntegerType())) \
        .withColumn('day', sqf_data['datestop'].substr(3, 2).cast(IntegerType())) \
        .withColumn('year', sqf_data['year'].cast(IntegerType())) \
        .withColumn('time_period', sqf_data['timestop'].substr(1, 2).cast(IntegerType()))
    
    if 2014 in years:
        for i in coln:
            sqf_data = sqf_data \
                .withColumn(i, when((sqf_data['year'] == 2014) & (sqf_data[i] == 1), 'Y')\
                            .when((sqf_data['year'] == 2014) & sqf_data[i].isNull(), 'N')\
                            .otherwise(sqf_data[i]))
    
    sqf_data = sqf_data \
        .withColumn('found_pistol', recode_yn_udf(sqf_data['pistol']))\
        .withColumn('found_rifle', recode_yn_udf(sqf_data['riflshot']))\
        .withColumn('found_assault', recode_yn_udf(sqf_data['asltweap']))\
        .withColumn('found_machinegun', recode_yn_udf(sqf_data['machgun']))\
        .withColumn('found_knife', recode_yn_udf(sqf_data['knifcuti']))\
        .withColumn('found_other', recode_yn_udf(sqf_data['othrweap']))\
        .withColumn('precinct', sqf_data['pct'].cast(IntegerType()))\
        .withColumn('additional_associating', recode_yn_udf(sqf_data['ac_assoc']))\
        .withColumn('additional_direction', recode_yn_udf(sqf_data['ac_cgdir']))\
        .withColumn('additional_report', recode_yn_udf(sqf_data['ac_rept']))\
        .withColumn('additional_evasive', recode_yn_udf(sqf_data['ac_evasv']))\
        .withColumn('additional_highcrime', recode_yn_udf(sqf_data['ac_incid']))\
        .withColumn('additional_investigation', recode_yn_udf(sqf_data['ac_inves']))\
        .withColumn('additional_proximity', recode_yn_udf(sqf_data['ac_proxm']))\
        .withColumn('additional_time', recode_yn_udf(sqf_data['ac_time']))\
        .withColumn('additional_sights', recode_yn_udf(sqf_data['ac_stsnd']))\
        .withColumn('additional_other', recode_yn_udf(sqf_data['ac_other']))\
        .withColumn('stopped_bulge', recode_yn_udf(sqf_data['cs_objcs']))\
        .withColumn('stopped_object', recode_yn_udf(sqf_data['cs_descr']))\
        .withColumn('stopped_casing', recode_yn_udf(sqf_data['cs_casng']))\
        .withColumn('stopped_clothing', recode_yn_udf(sqf_data['cs_lkout']))\
        .withColumn('stopped_desc', recode_yn_udf(sqf_data['cs_cloth']))\
        .withColumn('stopped_drugs', recode_yn_udf(sqf_data['cs_drgtr']))\
        .withColumn('stopped_furtive', recode_yn_udf(sqf_data['cs_furtv']))\
        .withColumn('stopped_lookout', recode_yn_udf(sqf_data['cs_vcrim']))\
        .withColumn('stopped_violent', recode_yn_udf(sqf_data['cs_bulge']))\
        .withColumn('stopped_other', recode_yn_udf(sqf_data['cs_other']))\
        .withColumn('inside', recode_io_udf(sqf_data['inout']))\
        .withColumn('observation_period', sqf_data['perobs'].cast(IntegerType()))\
        .withColumn('radio_run', recode_yn_udf(sqf_data['radio']))\
        .withColumn('location_housing', 
                    when(sqf_data['trhsloc'] == 'P', 'neither')\
                        .when(sqf_data['trhsloc'] == 'H', 'housing')\
                        .when(sqf_data['trhsloc'] == 'T', 'transit')\
                        .otherwise('neither'))\
        .withColumn('suspect_build',
                    when(sqf_data['build'] == 'H', 'heavy')\
                        .when(sqf_data['build'] == 'M', 'medium')\
                        .when(sqf_data['build'] == 'T', 'thin')\
                        .when(sqf_data['build'] == 'U', 'muscular')\
                        .when(sqf_data['build'] == 'Z', 'unknown')\
                        .otherwise(sqf_data['build']))\
        .withColumn('suspect_sex',
                    when(sqf_data['sex'] == 'M', 'male')\
                        .when(sqf_data['sex'] == 'F', 'female')\
                        .otherwise(sqf_data['sex']))
    
    sqf_data = sqf_data.withColumn('found_weapon', 
                               (sqf_data['found_pistol'] | 
                                sqf_data['found_rifle'] |
                                sqf_data['found_assault'] | 
                                sqf_data['found_machinegun'] |
                                sqf_data['found_knife'] | 
                                sqf_data['found_other']))

    sqf_data = sqf_data.filter(sqf_data['age'] != '**')
    sqf_data = sqf_data.filter(sqf_data['age'].isNotNull())
    sqf_data = sqf_data\
        .withColumn('suspect_age', 
                    sqf_data['age'].cast(IntegerType()))\
        .withColumn('suspect_height', 
                    sqf_data['ht_feet'] + (sqf_data['ht_inch'] / 12))\
        .withColumn('suspect_weight', sqf_data['weight'])
    
    sqf_data = sqf_data\
        .filter((sqf_data['suspect_age'] >= 5) & (sqf_data['suspect_age'] <= 100))\
        .filter(sqf_data['suspect_weight'] < 700)

    sqf_data = sqf_data.drop(*drop_column_names)
    sqf_data.show()

    print("--- Took: %s seconds ---\n" % (time.time()-start_time))

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('preprocessing').getOrCreate()

    # years = [2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016]
    years = [2008, 2009]
    main(spark, years)