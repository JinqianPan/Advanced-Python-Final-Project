import pandas as pd
import time

column_names = ['year', 'pistol', 'riflshot', 'asltweap', 'machgun', 
                    'knifcuti', 'othrweap', 'pct', 'trhsloc', 
                    'ac_assoc', 'ac_cgdir', 'ac_rept', 'ac_evasv', 'ac_incid', 
                    'ac_inves', 'ac_proxm', 'ac_time', 'ac_stsnd', 'ac_other',
                    'cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'cs_cloth', 
                    'cs_drgtr', 'cs_furtv', 'cs_vcrim', 'cs_bulge', 'cs_other', 
                    'age', 'build', 'sex', 'ht_feet', 'ht_inch', 'weight', 
                    'inout', 'radio', 'perobs', 'datestop', 'timestop']
    
coln = ["cs_objcs", "cs_descr", "cs_casng","cs_lkout", "cs_cloth", 
            "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other",
            "ac_rept", "ac_inves", "ac_proxm", "ac_evasv", "ac_assoc", 
            "ac_cgdir", "ac_incid", "ac_time", "ac_stsnd", "ac_other", 
            "pistol", "riflshot", "asltweap", "knifcuti", "machgun", 
            "othrweap"]

location_housing_recode_dict = {'P': 'neither', 
                                'H': 'housing', 
                                'T': 'transit'}

build_recode_dict = {'H': 'heavy', 'M': 'medium', 'T': 'thin', 
                        'U': 'muscular', 'Z': 'unknown'}

sex_recode_dict = {'M': 'male', 'F': 'female'}

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

def load_data(years: list):
    dataframes = []
    for year in years:
        filename = f'./data/{year}.csv'

        try: 
            this_data = pd.read_csv(filename, na_values=' ')
        except:
            this_data = pd.read_csv(filename, encoding= 'unicode_escape', na_values=' ')
    
        if year in range(2011, 2017):
            this_data = this_data.drop(columns=['forceuse'])
        if year in range(2013, 2017):
            this_data = this_data.rename(columns={'dettypCM': 'dettypcm', 
                                                  'lineCM': 'linecm', 
                                                  'detailCM': 'detailcm'})
            
        dataframes.append(this_data)
    sqf_data_full = pd.concat(dataframes, ignore_index=True)
    sqf_data = sqf_data_full.copy()

    return sqf_data

def recode_yn(f):
    f_new = f.replace({'N': 0, 'Y': 1})
    f_new = f_new.astype(bool)
    return f_new

def recode_io(f):
    f_new = f.replace({'O': 0, 'I': 1})
    f_new = f_new.astype(bool)
    return f_new

def main(years: list):
    start_time = time.time()

    sqf_data = load_data(years)

    sqf_data = sqf_data[column_names]
    sqf_data = sqf_data.dropna(subset=['timestop'])

    sqf_data['datestop'] = sqf_data['datestop'].apply(lambda x: '{0:0>8}'.format(x))
    sqf_data['timestop'] = sqf_data['timestop'].apply(lambda x: '{0:0>4}'.format(x))

    sqf_data['month'] = sqf_data['datestop'].str[:2].astype(int)
    sqf_data['day'] = sqf_data['datestop'].str[2:4].astype(int)
    sqf_data['year'] = sqf_data['year'].astype(int)

    sqf_data['time_period'] = sqf_data['timestop'].str[:2].astype(int)
    
    if 2014 in years:
        for i in coln:
            sqf_data.loc[(sqf_data['year'] == 2014) & (sqf_data[i] == 1), i] = 'Y'
            sqf_data.loc[(sqf_data['year'] == 2014) & (sqf_data[i].isna()), i] = 'N'
    
    sqf_data = sqf_data.assign(
        found_pistol = recode_yn(sqf_data['pistol']),
        found_rifle = recode_yn(sqf_data['riflshot']),
        found_assault = recode_yn(sqf_data['asltweap']),
        found_machinegun = recode_yn(sqf_data['machgun']),
        found_knife = recode_yn(sqf_data['knifcuti']),
        found_other = recode_yn(sqf_data['othrweap']),
        precinct=pd.factorize(sqf_data['pct'])[0]+1,
        additional_associating = recode_yn(sqf_data['ac_assoc']),
        additional_direction = recode_yn(sqf_data['ac_cgdir']),
        additional_report = recode_yn(sqf_data['ac_rept']),
        additional_evasive = recode_yn(sqf_data['ac_evasv']),
        additional_highcrime = recode_yn(sqf_data['ac_incid']),
        additional_investigation = recode_yn(sqf_data['ac_inves']),
        additional_proximity = recode_yn(sqf_data['ac_proxm']),
        additional_time = recode_yn(sqf_data['ac_time']),
        additional_sights = recode_yn(sqf_data['ac_stsnd']),
        additional_other = recode_yn(sqf_data['ac_other']),
        stopped_bulge = recode_yn(sqf_data['cs_objcs']),
        stopped_object = recode_yn(sqf_data['cs_descr']),
        stopped_casing = recode_yn(sqf_data['cs_casng']),
        stopped_clothing = recode_yn(sqf_data['cs_lkout']),
        stopped_desc = recode_yn(sqf_data['cs_cloth']),
        stopped_drugs = recode_yn(sqf_data['cs_drgtr']),
        stopped_furtive = recode_yn(sqf_data['cs_furtv']),
        stopped_lookout = recode_yn(sqf_data['cs_vcrim']),
        stopped_violent = recode_yn(sqf_data['cs_bulge']),
        stopped_other = recode_yn(sqf_data['cs_other']),
        inside = recode_io(sqf_data['inout']),
        observation_period = sqf_data['perobs'],
        radio_run = recode_yn(sqf_data['radio']),
        location_housing = sqf_data['trhsloc'].replace(location_housing_recode_dict).fillna('neither'),
        suspect_build = sqf_data['build'].replace(build_recode_dict),
        suspect_sex = sqf_data['sex'].replace(sex_recode_dict)
        )
    
    sqf_data['found_weapon'] = (sqf_data['found_pistol'] | 
                                sqf_data['found_rifle'] |
                                sqf_data['found_assault'] | 
                                sqf_data['found_machinegun'] |
                                sqf_data['found_knife'] | 
                                sqf_data['found_other'])

    sqf_data = sqf_data.drop(sqf_data[sqf_data['age'] == '**'].index)
    sqf_data = sqf_data.dropna(subset=['age'])
    sqf_data['suspect_age'] = sqf_data['age'].astype(int)
    sqf_data = sqf_data.loc[(sqf_data['suspect_age'] >= 5) & 
                            (sqf_data['suspect_age'] <= 100)]
    sqf_data = sqf_data.dropna(subset=['suspect_age'])

    sqf_data['suspect_height'] = sqf_data['ht_feet'] + (
        sqf_data['ht_inch'] / 12)
    sqf_data['suspect_weight'] = sqf_data['weight']

    # replace suspect.weight >= 700 with NA
    sqf_data = sqf_data.loc[sqf_data['suspect_weight'] < 700]

    sqf_data = sqf_data.drop(columns=drop_column_names)
    print("--- Took: %s seconds ---\n" % (time.time()-start_time))

    sqf_data.to_csv('./data/sqf_data.csv', index=False)


if __name__ == "__main__":
    years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    main(years)