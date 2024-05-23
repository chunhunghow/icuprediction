
import pandas as pd
import numpy as np
import os, re, sys, math
import datetime as dt
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from functools import partial
import warnings
from tqdm import tqdm
from typing import Optional
import pickle
import time
from utils.utils import SanityCheck


    
class ICUAdm():
    '''
    Represent each admission of an encounter.

    Some attributes:
    problist - Problem list of the patient of this admission.
    fs_att - Flowsheet data in dictionary during this admission.
    orders - Orders recorded during this admission.
    cci_score - CCI of the patient.
    '''
    
    def __init__(self , enc_id, adm_time:Optional = None):
        self.enc_id = enc_id
        self.adm_time = adm_time
        self.__get_icu_info()
        self.__get_attr()
        self.__get_score()
        

        
            
    def __get_icu_info(self):
        '''
        __get_icu_info() 
        retrieves data form flowsheet (year) and patient demographic data, declared in working notebook which loaded the dataset.

        runs class functions data as a dictionary data structure
        as well as expand the data into hourly records

        '''
#         fs = globals().get('fs')
#         pat = globals().get('pat_cohort' if 'pat_cohort' in globals() else 'pat') ##changed from pat
#         pro = globals().get('pro')
#         odr = globals().get('odr')
#         sur = globals().get('sur')
#         sed = globals().get('sed')
#         icd_codes_dict = globals().get('icd_codes_dict')
#         chronic_prob = globals().get('chronic_prob')

        
        assert self.enc_id in fs['Encounter ID'].unique()
            
        if self.adm_time:
            self.enc_info = pat[(pat['PAT_ENC_CSN_ID'] == self.enc_id) & (pat['ICU Admission Time']==self.adm_time)].iloc[0,:18].to_dict()
        else : self.enc_info = pat[pat['PAT_ENC_CSN_ID'] == self.enc_id].sort_values(by='ICU Admission Time').iloc[0,:18].to_dict() #sort by icu adm time and take the first entry
        self.icu_right_limit = self.enc_info['ICU Discharge Time'].date()  + dt.timedelta(days=1) #buffer 1 day for flowsheet value
        self.icu_left_limit = self.enc_info['ICU Admission Time'].date()
        self.fs = fs[(fs['Encounter ID'] == self.enc_id) & (fs['Flowsheet Recorded Time'].dt.date < self.icu_right_limit) & (fs['Flowsheet Recorded Time'].dt.date >= self.icu_left_limit)] # only flowsheet strictly before the next day
        self.fs = self.fs.drop_duplicates().sort_values(by='Flowsheet Recorded Time')
#         self.pat = pat[pat['PAT_ENC_CSN_ID'] == self.enc_id].drop_duplicates().sort_values(by='INP_ADM_DATE')
        self.pat = pat
        
        surg_discharge, surg_type = sur[sur['PAT_ENC_CSN_ID'] == self.enc_id]['OUT_OR_DTTM'].to_numpy() , sur[sur['PAT_ENC_CSN_ID'] == self.enc_id] ['NAME'].to_numpy()
        self.surg_discharge = surg_discharge
        # compare all surg out time, if icu adm is wihtin 3 hours after surg, get surg type
        if len(surg_discharge) > 0:
            for i,t in enumerate(surg_discharge): 
                if (dt.timedelta(hours=0) <=  self.enc_info['ICU Admission Time'] - t < dt.timedelta(hours=3)):
                    self.enc_info['Surg_prior_ICU'] = surg_type[i] if not isinstance(surg_type[i],float) else 'None'
                    
        
        if 'Surg_prior_ICU' not in self.enc_info.keys(): self.enc_info['Surg_prior_ICU'] = 'None' #no info about this patient
        
       
        
    def __get_attr(self):
        
        self.__get_hourly()
        self.__get_problemlist()
        self.__get_sed()
        self.__get_fs_att()
        self.__get_odr()
        self.__get_first_intubation_datetime()
        self.__get_ventilated_duration()
#         self.__get_GCS_score()
        
    def __get_score(self):
        self.__get_sofa()
        self.__get_apache()
        

    def __get_fs_att(self):
        '''
        collate the records by rounding to the nearest hour base on the flowsheet recorded time
        '''
        self.fs_att = {}
        for col in self.fs['Flowsheet Name'].unique():
            data = self.__arrange_hourly(self.fs[self.fs['Flowsheet Name'] == col][['Flowsheet Recorded Time','Flowsheet Value']])
            data = data.fillna(value=np.nan) # to replace None to NaN
            data = pd.merge(pd.DataFrame(zip(self.hourly, [1]*len(self.hourly)), columns=['nearest_hour','temp']), data, on='nearest_hour',how='left')
            self.fs_att[col] = dict(data[['nearest_hour','Flowsheet Value']].to_numpy())
            
            
        for k in ['R MAP','TEMPERATURE','R FIO2','R CPN GLASGOW COMA SCALE BEST MOTOR RESPONSE','R CPN GLASGOW COMA SCALE SCORE','PULSE']:
            if k not in self.fs_att.keys(): # missing data
                subs_value = fs[fs['Flowsheet Name'] == k]['Flowsheet Value'].mode().values[0]
                self.fs_att[k] = dict(zip(self.hourly,[subs_value]*len(self.hourly)))
                
                    
    def __get_hourly(self):
        '''
        expands the data into hourly format and round each flowsheet recorded item to the nearest hour
        '''
        dif_hour = np.ceil((self.enc_info['ICU Discharge Time']  - self.enc_info['ICU Admission Time']).total_seconds() / 60 / 60).astype(int)
        starttime = hour_round(self.enc_info['ICU Admission Time']) 
#         self.__hourly = [starttime - timedelta(hours=i) for i in range(12,0,-1)] + [starttime + timedelta(hours=i) for i in range(dif_hour )]
        self.hourly =  [starttime + timedelta(hours=i) for i in range(dif_hour )]

    
    def __get_problemlist(self):
        '''
        ICD codes
        '''
        self.problist = dict(pro[pro['Encounter ID'] == self.enc_id][['DX_NAME','CURRENT_ICD10_LIST']].values)
        self.cci_score = 0
        if len(self.problist.values()) == 0: self.cci_diag = None
        for codes in self.problist.values():
            if type(codes) == float: continue
            self.cci_score += self.__get_cci_score(codes)
        
        self.enc_info['CCI Score'] = self.cci_score
        self.enc_info['CCI Diagnosis'] = self.cci_diag
        self.enc_info['Chronic'] = True if (any([k in chronic_prob for k in self.problist.keys()])) else False
    
    def __get_sed(self):
        '''
        sedation records
        '''
        self.sed = sed[(sed['Encounter ID'] == self.enc_id) & (sed['Time of Administration'].dt.date >= self.icu_left_limit) & (sed['Time of Administration'].dt.date < self.icu_right_limit)]
        
        if self.sed.shape[0] != 0:
            self.sed['round_hour'] = self.sed['Time of Administration'].apply(lambda x: hour_round(x))
            self.sed['tup'] = self.sed.apply(lambda row: tuple(row[['Medication Name', 'Dose Amount','Dose Unit']].values),1)
            self.sed = self.sed.groupby('round_hour').apply(lambda g: list(g['tup'])).to_dict()
        else: #not sedated
            self.sed = {}
    
    def __get_odr(self):
        '''
        blood test 
        '''
        self.odr = odr[(odr['Encounter ID']== self.enc_id) & (odr['Order Result Time'].dt.date >= self.icu_left_limit) & (odr['Order Result Time'].dt.date < self.icu_right_limit)]
        if self.odr.empty: return
        self.odr['round_hour'] = self.odr['Order Result Time'].apply(lambda x: hour_round(x))
        self.odr['component_tup'] = self.odr.apply(lambda row: tuple(row[['Component Name','Order Numeric Value','Unit']].values),1)
        self.odr = self.odr.groupby(['round_hour']).apply(lambda g: list(g['component_tup'])).to_dict()
    
    def __arrange_hourly(self, data):
        '''For flowsheet'''
        data = data.sort_values(by='Flowsheet Recorded Time')
        data['nearest_hour'] = data['Flowsheet Recorded Time'].apply(lambda x: hour_round(x))
        return data.groupby(['nearest_hour']).agg({'Flowsheet Value':'last'})
        
    def __get_cci_score(self, strings):
        cci_score = 0
        cci_diag = []
        arr = [re.sub(' ','',c) for c in re.sub('\.','',strings).split(',')]
        for code in arr:
            try:
                cci_score += cciscoremap[code]
                if code != []:
                    cci_diag.append(code)
            except KeyError:
                '''
                if keyError, it means the icd code is not inside the CCI's list of icd codes. 
                hence, icd code is not CCI, we can pass this loop
                '''
                pass
        
        self.cci_diag = cci_diag
        return cci_score

    def __get_ventilated_duration(self):
        '''Duration of ventilation according to first and last Endotracheal flowsheet time'''
        if 'R OXYGEN DEVICE' in self.fs_att.keys(): 
            tube_df = pd.DataFrame.from_dict(self.fs_att['R OXYGEN DEVICE'].items())
            tube_df.columns = ['time','values']
            tube_df['values'] = tube_df['values'].fillna(method='bfill')
            self.enc_info['mv_duration'] = np.sum(tube_df['values'] == 'Endotracheal tube')
#             tube_time = np.where(np.array(list(tube_df.values())) == 'Endotracheal tube')[0]
#             self.enc_info['mv_duration'] = tube_time[-1] - tube_time[0]
        else: self.enc_info['mv_duration'] = np.nan
        
        
    def __get_first_intubation_datetime(self):
        '''
        Check for first occurrance of MV which would determine intubation
        If cant find then take admission time
        '''
        if 'R RT TYPE OF VENTILATOR' not in self.fs_att.keys():
            self.enc_info['First Intubated'] = self.enc_info['ICU Admission Time']
            return
            
        for k,v in self.fs_att['R RT TYPE OF VENTILATOR'].items():
            if k >= self.enc_info['ICU Admission Time'] and\
            k <= self.enc_info['ICU Discharge Time'] and\
            str(v) in ('Mechanical Ventilator'):
                self.enc_info['First Intubated'] = k
                break
    
    def __get_GCS_score(self): 
        ''' 
        Gets GCS Motor score and GCS Scale Score
        '''
        for k,v in self.fs_att['R CPN GLASGOW COMA SCALE BEST MOTOR RESPONSE'].items():
            if k >= self.enc_info['ICU Admission Time'] and\
            k <= self.enc_info['ICU Discharge Time'] and\
            not math.isnan(v):
                self.enc_info['GCS Motor Score'] = (k,v)
                break
                
        for k,v in self.fs_att['R CPN GLASGOW COMA SCALE SCORE'].items():
            if k >= self.enc_info['ICU Admission Time'] and\
            k <= self.enc_info['ICU Discharge Time'] and\
            not math.isnan(v):
                self.enc_info['GCS Score'] = (k,v)
                break
    
    
    def __get_sofa(self):   

        
        epinephrine= ['EPINEPHRINE (ADRENALINE) 1MG/10ML INJECTION','EPINEPHRINE (ADRENALINE) INJECTION (NACL 0.9, D5) (COMPONENT: 1MG/ML INJECTION)','(PUMP) EPINEPHRINE (ADRENALINE) 1MG/ML INJECTION (INTRAMUSCULAR, SUBCUTANEOUS)']
        norepinephrine = ['(PUMP) NOREPINEPHRINE (NORADRENALINE) (LEVOPHED) 4MG/4ML INJECTION']
        dop = ['(PUMP) DOPAMINE HCL 200MG/5ML INJECTION']
        
               
        
        #sedation
        def sed_process(tup):
            '''
            make sure all dose amount is not NA
            '''
            sed = []
            for t in tup:
                if math.isnan(t[1]): # none type
                    continue 
                if t[0] in dop:
                    if 'mcg' in t[-1]:
                        dosage = t[1] / 1000 # to mg
                    elif 'mg' in t[-1]:
                        dosage = t[1]
                    else:
                        #none type
                        dosage = t[1] 
                    
                    sed += [('dop',dosage)]
                
                elif t[0] in epinephrine:
                    if 'mcg' in t[-1]:
                        dosage = t[1] / 1000 # to mg
                    elif 'mg' in t[-1]:
                        dosage = t[1]
                    else:
                        #none type
                        dosage = t[1] 
                        
                    sed += [('epp',dosage)]
                    
                elif t[0] in norepinephrine:
                    if 'mcg' in t[-1]:
                        dosage = t[1] / 1000 # to mg
                    elif 'mg' in t[-1]:
                        dosage = t[1]
                    else:
                        #none type
                        dosage = t[1] 
                        
                    sed += [('norepp',dosage)]
                
                else:
                    continue

            return sed
                    
        
        def pf_ratio_score(tup, raw):
            
            
            # tup like (pf_ratio, value, unit)
            mmhg = tup[1]
            if raw: return mmhg 
                
                
            if math.isnan(mmhg):
                return None

            if mmhg < 100:
                return 4
            elif mmhg < 200:
                return 3
            elif mmhg < 300:
                return 2
            elif mmhg < 400:
                return 1
            else:
                return 0
            
        


        def platelet_score(tup,raw):
            # tup like (platelet, value, unit)
            count = tup[1]
            
            if raw: return count
            
            if math.isnan(count):
                return None

            if count <20:
                return 4
            elif count <50:
                return 3
            elif count < 100:
                return 2
            elif count < 150:
                return 1
            else:
                return 0
            
            
            
        def bilirubin_score(tup,raw):
            # tup like (bilirubin, value, unit)
            mmol = tup[1]
            
            if raw: return mmol
            
            if math.isnan(mmol):
                return None
            if mmol >204:
                return 4
            elif mmol > 101:
                return 3
            elif mmol > 32:
                return 2
            elif mmol >= 20:
                return 1
            else:
                return 0 
            
            
        def creatinine_score(tup,raw):
            # tup like (creatinine, value, unit)
            mmol = tup[1]
            
            if raw: return mmol
            
            if math.isnan(mmol):
                return None
            if mmol > 440:
                return 4
            elif mmol > 299:
                return 3
            elif mmol > 170:
                return 2
            elif mmol >= 110:
                return 1
            else:
                return 0
            
            
        def gcs_score(score, raw):
            
            if raw: return score
            
            if type(score) == str:
                score = float(score)
                
            if math.isnan(score):
                return None
            if score <6:
                return 4
            elif score < 10:
                return 3
            elif score < 13:
                return 2
            elif score < 15:
                return 1
            else:
                return 0
            
                
        
        
        def hypotension_score(self,time,tups,raw):
            if tups == [] : tups = None
            if tups is not None:
                sedation_order = dict(tups)

                if 'dop' in sedation_order.keys():
                    dose_amount = sedation_order['dop']

                    if raw : return dose_amount

                    if dose_amount >15:
                        return 4
                    elif dose_amount > 5:
                        return 3
                    else: 
                        return 2
                else:
                    if ('epp' in sedation_order.keys()) or ('norepp' in sedation_order.keys()):
                        dose_amount = sedation_order['epp'] if ('epp' in sedation_order.keys()) else sedation_order['norepp']
                        if raw: return dose_amount
                        if dose_amount > 0.1:
                            return 4
                        else:
                            return 3


            else:
                #check if MAP is recorded at time T
                if time in self.fs_att['R MAP'].keys(): 
                    MAP = self.fs_att['R MAP'][time]
                    MAP = float(MAP)
                else: MAP = None #not recorded
                        
                if raw: return MAP # sofa_comp return raw MAP

                if math.isnan(MAP): # if MAP is nan, then same as not recorded
                    return None
                
                if MAP <70:
                    return 1
                else:
                    return 0 #no hypotension

            
            
            
            
        pf_ratio = {}
        platelet_count = {}
        bilirubin = {}
        creatinine = {}
        sedation = {}
        
        

        
        # for orders 
        ##TODO need to change name , didnt expect clarity to change comp name (JHC) (NUHS)
        for k,v in self.odr.items(): 
            componentName = [tup[0] for tup in v]
            f = lambda x: [comp for comp in componentName if x in comp]
            


            if any(f('PO2/FIO2')):
                pf_ratio[k] = v[np.where(np.array(componentName) == f('PO2/FIO2')[0])[0][-1]]
                
            if any(f('PLATELET COUNT')):
                platelet_count[k] = v[np.where(np.array(componentName) == f('PLATELET COUNT')[0])[0][-1]]
        
            if any(f('BILIRUBIN, TOTAL')):
                bilirubin[k] = v[np.where(np.array(componentName) == f('BILIRUBIN, TOTAL')[0])[0][-1]]
                
            if any(f('CREATININE, SERUM')):
                creatinine[k] = v[np.where(np.array(componentName) == f('CREATININE, SERUM')[0])[0][-1]]

        #for sedation      
        for k,v in self.sed.items():
            sedation[k] = sed_process(v)

        self.sedation = sedation
        
        gcs = SanityCheck(self.fs_att['R CPN GLASGOW COMA SCALE SCORE'], float)

        

        # arrange sofa components by each hours
        
        
        def check_key(var_name,var,func,time, raw=False):


            
            if var_name != 'sedation':
                if time in var.keys():
                    return func(var[time],raw)
                else:
                    return None
                
            else: #hypotension variable take in R MAP also
                if time in var.keys():
                    return func(time,var[time],raw)
                else: #sedation not recorded, find MAP
                    return func(time,None, raw)
                
        
            
        self._sofa_comp = {}
        self._sofa_score = {}
        
        var = {'pf_ratio':pf_ratio_score, 
               'platelet_count':platelet_score,
               'bilirubin':bilirubin_score,
               'creatinine':creatinine_score, 
               'sedation':partial(hypotension_score,self), 
               'gcs':gcs_score}

        

        
        for t in self.hourly:
            for v,f in var.items():
                if t not in self._sofa_score.keys():
                    self._sofa_score[t] = [check_key(v,locals().get(v),f,t)]
                    self._sofa_comp[t] = [check_key(v,locals().get(v),f,t, True)]
                else:
                    self._sofa_score[t] += [check_key(v,locals().get(v),f,t)]
                    self._sofa_comp[t] += [check_key(v,locals().get(v),f,t, True)]

        
        
       
    
    def __get_apache(self):
        

        temperature = SanityCheck(self.fs_att['TEMPERATURE'], float)
        MAP = SanityCheck(self.fs_att['R MAP'], float)
        gcs = SanityCheck(self.fs_att['R CPN GLASGOW COMA SCALE SCORE'], float) #replaced in SOFA if none
        pulse = SanityCheck(self.fs_att['PULSE'], float)
        fio2 = SanityCheck(self.fs_att['R FIO2'], float)
        age = self.enc_info['AGE_AT_ADMISSION']
        surg_type = self.enc_info['Surg_prior_ICU']
        chronic = self.enc_info['Chronic']
        self.chronic = chronic
        
        self.death_before_disc = list(pulse.values())[-1] == 0
        if self.death_before_disc:
            self.death_time = list(pulse.values()).index(0)
            self.death_time = list(pulse.keys())[self.death_time]
        else: self.death_time = None

        
        def temperature_score(fahr,raw):
            celc = (fahr - 32) * 5 / 9
            if raw: return celc
            if math.isnan(celc) : return None
            if (celc >= 41) | (celc <= 29.9):
                return 4
            elif (39 <= celc <= 40.9) | (30 <= celc <= 31.9):
                return 3
            elif (32 <= celc <= 33.9):
                return 2
            elif (38 <= celc <= 38.4) | (32 <= celc <= 33.9):
                return 1
            else:
                return 0
            
        def map_score(mmhg,raw):
            if raw: return mmhg
            if math.isnan(mmhg): return None
            if (mmhg <= 49) | (mmhg >= 160):
                return 4
            elif (130 <= mmhg <= 159):
                return 3
            elif (110 <= mmhg <= 129) | (50 <= mmhg <= 69):
                return 2
            else:
                return 0
            
        def heart_rate(rate,raw):
            if raw: return rate
            
            if math.isnan(rate) : return None
            if (rate <= 40) | (rate >= 180):
                return 4
            elif (140 <= rate <= 179) | (40 <= rate <= 54):
                return 3
            elif (110 <= rate <= 139) | (55 <= rate <= 69):
                return 2
            else: # should not be 0
                return 0 
        
        def sodium_score(tup,raw):
            mM = tup[1]
            
            if raw: return mM
            
            if math.isnan(mM):
                return None
            if (mM >= 180) | (mM <= 110):
                return 4
            elif (160 <= mM <= 179) | (111 <= mM <= 119):
                return 3
            elif (155 <= mM <= 159) | (120 <= mM <= 129):
                return 2
            elif (150 <= mM <= 154):
                return 1
            else:
                return 0
        
        def potassium_score(tup,raw):
            mM = tup[1]
            
            if raw: return mM
            
            if math.isnan(mM):
                return None
            if (mM >= 7) | (mM < 2.5):
                return 4
            elif (6 <= mM <= 6.9):
                return 3
            elif (2.5 <= mM <= 2.9):
                return 2
            elif (5 <= mM <= 5.9) | (3 <= mM <= 3.4):
                return 1
            else:
                return 0
            
        
        
        def creatinine_score(tup,raw):
            # tup like (creatinine, value, unit)
            mmol = tup[1]
            if raw: return mmol
            mgl = mmol * 0.232019
            
            if math.isnan(mgl):
                return None
            if mgl >= 35:
                return 4
            elif 20 <= mgl <= 34 :
                return 3
            elif (15<= mgl <= 19) | (mgl < 6):
                return 2
            else:
                return 0
            
        def hematocrit_score(tup,raw):
            
            perc = tup[1]
            if raw: return perc
            
            if math.isnan(perc):
                return None
            if (perc >= 60) | (perc < 20):
                return 4
            elif (50 <= perc <= 59.9) | (20 <= perc <= 29.9):
                return 2
            elif 46 <= perc <= 46.9:
                return 1
            else:
                return 0
            
        def whitebloodcell_score(tup,raw):
            count = tup[1]
            if raw: return count
            
            if math.isnan(count):
                return None
            if (count >= 40) | (count < 1):
                return 4
            elif (20 <= count <= 39.9) | (1 <= count <= 2.9):
                return 2
            elif 15 <= count <= 19.9:
                return 1
            else:
                return 0
        
        def ph_score(tup,raw):
            val = tup[1]
            if raw: return val
            
            if math.isnan(val):
                return None
            if (val >= 7.7) | (val < 7.15):
                return 4
            elif (7.6 <= val <= 7.69) | (7.15 <= val <= 7.24):
                return 3
            elif (7.25 <= val <= 7.32):
                return 2
            elif (7.5 <= val <= 7.59):
                return 1
            else:
                return 0
            
        
        def gcs_score(gcs,raw):
            if raw: return gcs
            
            if math.isnan(gcs):
                return None
            score = (15 - gcs) if (gcs > 3) else 12
            return score
            
            
        def age_score(raw):
            #takes global var age
            
            if raw: return age
            
            if age >= 75:
                return 6
            elif 65 <= age <= 74:
                return 5
            elif 55 <= age <= 64:
                return 3
            elif 45 <= age <= 54:
                return 2
            else:
                return 0
            
            
        def chronic_health_score(raw):
            if raw : return surg_type
            
            
            if chronic:
                if surg_type == 'Elective':
                    return 2
                elif 'P' in surg_type:
                    return 5 #emergency
                else: #either No info or nan
                    return 0
            else:
                return 0
            


        def Aa_gradient_score(args,raw):
            def Aa_gradient(*args):
                fio2,PaCO2, PaO2 = args
                PAO2 = (760 - 47) * fio2/100 - PaCO2 / 0.8
            
                return PAO2 - PaO2
        
            fio2,PaCO2,PaO2 = args
            if isinstance(fio2, tuple): fio2 = fio2[1] * 100 #from orders, the other directly from flowsheet
            if isinstance(PaCO2, tuple): PaCO2 = PaCO2[1]
            if isinstance(PaO2, tuple): PaO2 = PaO2[1]
                
            if fio2 >= 50:
                aagrad = Aa_gradient(fio2,PaCO2,PaO2)
                if raw: return f'{fio2},{PaCO2},{PaO2}' ##TODO
                
                if aagrad >= 500:
                    return 4
                elif 350 <= aagrad <= 499:
                    return 3
                elif 200 <= aagrad <= 349:
                    return 2
                else: #<200
                    return 0
            else:

                if raw: return f'{fio2},{PaCO2},{PaO2}'
                if (math.isnan(PaO2) | math.isnan(fio2)): return None
                
                if PaO2 < 55:
                    return 4
                elif 55 <= PaO2 <=60:
                    return 3
                elif 61 <= PaO2 <= 70:
                    return 1
                else: # > 70
                    return 0
            
            
        sodium = {}
        potassium = {}
        creatinine = {}
        hematocrit = {}
        wbc = {}
        ph = {}
        PaCO2 = {}
        PaO2 = {}
        
        
        
        for k,v in self.odr.items(): 
            componentName = [tup[0] for tup in v]
            f = lambda x: [comp for comp in componentName if x in comp]
                
            if any(f('POCT SODIUM')):
                sodium[k] = v[np.where(np.array(componentName) == f('POCT SODIUM')[0])[0][-1]]
                
            if any(f('SODIUM, SERUM')):
                sodium[k] = v[np.where(np.array(componentName) == f('SODIUM, SERUM')[0])[0][-1]] #sodium serum will overwrite POCT SODIUM if same hour
                
                
            if any(f('POCT POTASSIUM')):
                potassium[k] = v[np.where(np.array(componentName) == f('POCT POTASSIUM')[0])[0][-1]]
            
            if any(f('POTASSIUM, SERUM')):
                potassium[k] = v[np.where(np.array(componentName) == f('POTASSIUM, SERUM')[0])[0][-1]] #potassium serum will overwrite POCT POTASSIUM if same hour
            
            if any(f('CREATININE, SERUM')):
                creatinine[k] = v[np.where(np.array(componentName) == f('CREATININE, SERUM')[0])[0][-1]]

            if any(f('HAEMATOCRIT')):
                hematocrit[k] = v[np.where(np.array(componentName) == f('HAEMATOCRIT')[0])[0][-1]]
            
            if any(f('WHITE BLOOD CELL COUNT')):
                wbc[k] = v[np.where(np.array(componentName) == f('WHITE BLOOD CELL COUNT')[0])[0][-1]]
            
            if any(f('POCT PH')):
                ph[k] = v[np.where(np.array(componentName) == f('POCT PH')[0])[0][-1]]
                
            if any(f('POCT PCO2')):
                PaCO2[k] = v[np.where(np.array(componentName) == f('POCT PCO2')[0])[0][-1]]
                
            if any(f('POCT PO2 ')):
                PaO2[k] = v[np.where(np.array(componentName) == f('POCT PO2 ')[0])[0][-1]]
              
            if any(f('POCT FIO2,')):
                # overwrite the value if filled by flowsheet
                fio2[k] = v[np.where(np.array(componentName) == f('POCT FIO2,')[0])[0][-1]] #orders are within 1, aagrad use /100
                

#         self.locals = locals()

        
        var = {'temperature':temperature_score,'MAP':map_score,
               'pulse':heart_rate,'ph':ph_score,'sodium':sodium_score,
               'potassium':potassium_score,'creatinine':creatinine_score,
              'hematocrit' : hematocrit_score, 'wbc':whitebloodcell_score,
                'gcs':gcs_score, 'age':age_score,
               'fio2,PaCO2,PaO2':Aa_gradient_score, #TODO , separate it so interpolate better
               'surg_type' : chronic_health_score
              }
        

        
        def check_key(var_name,var,func,time, raw=False):
            # rule based checking all variables, raw : return raw score
            
            
            # for static user value
            if var_name == 'age':
                return func(raw)
            
            if var_name == 'surg_type':
                return func(raw)
            
            # for non static value
            if type(var) == list:
                try:

                    return func([v[time] for v in var],raw)
                    
                except KeyError:
                    return None
                
                
                
            if time in var.keys():
                
                return func(var[time],raw)
            else:
                return None
        
        
        
        self._apache_comp = {}
        self._apache_score = {}
        
        lc = locals().copy() #all variables stored here
        for t in self.hourly:
            for v,f in var.items(): #v : variable name, f : function
                
                get_variable = lc.get(v) if (',' not in v) else [lc.get(v_i) for v_i in v.split(',')]

                try: get_variable = float(get_variable)
                except: pass
                
                if t not in self._apache_score.keys():
                    self._apache_score[t] = [check_key(v,get_variable,f,t)]
                    self._apache_comp[t] = [check_key(v,get_variable,f,t,True)]
                else:
                    self._apache_score[t] += [check_key(v,get_variable,f,t)]
                    self._apache_comp[t] += [check_key(v,get_variable,f,t,True)]
        
        
        
        
        self.aagrad_func = Aa_gradient_score
    
    @property
    def sofa_score(self):
        data = self._sofa_score
        data = pd.DataFrame(data).T.fillna(value=np.nan)
        data.columns = ['PF ratio', 'Platelet Count', 'Bilirubin', 'Creatinine', 'Hypotension', 'GCS']
#         data['SOFA'] = data.sum(axis=1)
        start_index = [t> self.enc_info['ICU Admission Time'] for t in data.index].index(True)
        return data.iloc[start_index:self.death_time ,:]
    
    @property
    def apache_score(self):

        data = self._apache_score
        data = pd.DataFrame(data).T
        data.columns = ['Temperature','MAP','Pulse','PH','Sodium','Potassium','Creatinine','Haematocrit','White Blood Cell','GCS','Age','A-a gradient','Chronic Status']
#         data['APACHE'] = data.sum(axis=1)
        start_index = [t> self.enc_info['ICU Admission Time'] for t in data.index].index(True)
        return data.iloc[start_index:self.death_time ,:]
    
    
    
    
    @property
    def sofa_comp(self):
        data = self._sofa_comp
        data = pd.DataFrame(data).T.fillna(value=np.nan)
        data.columns = ['PF ratio', 'Platelet Count', 'Bilirubin', 'Creatinine', 'Hypotension', 'GCS']
        data['hosp_after_icu'] = self.enc_info['hosp_after_icu']
        data['adm_idx'] = self.enc_info['adm_idx']
        data['mortality'] = int('death' in self.enc_info['DISCHARGE_DISPOSITION'].lower())
        start_index = [t> self.enc_info['ICU Admission Time'] for t in data.index].index(True)
        return data.iloc[start_index:self.death_time ,:]
    
    @property
    def apache_comp(self):

        data = self._apache_comp
        data = pd.DataFrame(data).T.fillna(value=np.nan)
        data.columns = ['Temperature','MAP','Pulse','PH','Sodium','Potassium','Creatinine','Haematocrit','White Blood Cell','GCS','Age','A-agrad/fio2/pco2,po2','Surg Prior ICU']
        data['Chronic'] = self.chronic
        
        
        start_index = [t> self.enc_info['ICU Admission Time'] for t in data.index].index(True)
        return data.iloc[start_index:self.death_time ,:]
    

    
def hour_round(time):
    return (time.replace(second=0, microsecond=0,minute=0,hour = time.hour) + timedelta(hours = time.minute//30))
   
    
timestamp = lambda timestr: datetime.strptime(timestr,'%Y-%m-%d %H:%M:%S') 




class Encounter:
    def __init__(self,enc_id, admission_idx=None):
        pat = globals().get('pat_cohort' if 'pat_cohort' in globals() else 'pat')
        self.encs = []
        adms = pat[pat['PAT_ENC_CSN_ID'] == enc_id].sort_values(by='ICU Admission Time')
        if admission_idx is not None:
            self.encs = [ICUAdm(enc_id, adms.iloc[admission_idx]['ICU Admission Time'])]
        else:
            for i in range(adms.shape[0]):
                self.encs += [ICUAdm(enc_id, adms.iloc[i]['ICU Admission Time'])]
    def __len__(self):
        return len(self.encs)
    @property
    def enc_info(self):
        return [e.enc_info for e in self.encs]
            
