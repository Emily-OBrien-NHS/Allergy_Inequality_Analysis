import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import scipy.stats.distributions as dist
import itertools
import seaborn as sns
import textwrap as tw
import os
import math
import matplotlib.pyplot as plt
file_path = 'C:/Users/obriene/Projects/Allergy/Inequalities'
# =============================================================================
#     #Read in and tidy data
# =============================================================================
sdmart_engine = create_engine('mssql+pyodbc://@SDMartDataLive2/InfoDB?'\
                              'trusted_connection=yes&driver=ODBC+Driver+17'\
                              '+for+SQL+Server')
allergy_sql = """SET NOCOUNT ON 
DECLARE    @startdate AS DATETIME
DECLARE   @enddate AS DATETIME
SET    @startdate ='01-JAN-2024'
SET    @enddate = GETDATE()

SELECT vwref.pasid, vwref.patnt_refno, --for joins to patients table
	   vwref.refrl_refno, --used for linking to activity etc
	   vwref.recvd_dttm AS [Referral Date],
	   proca.desc_ident AS [Referral Clinician],
	   sor.[description] AS [Referral Source],
	   OUTPAT.[start_dttm] AS [Date Seen],
	   DATEDIFF(DAY, vwref.recvd_dttm, OUTPAT.start_dttm) AS [Days Wait to be Seen],
       OUTPAT.[visit_desc] AS [Appoinment Details],
       OUTPAT.[pat_age_at_appt] AS [Patient Age at Appt],
	   ROW_NUMBER() OVER (PARTITION BY vwref.patnt_refno, vwref.refrl_refno ORDER BY OUTPAT.start_dttm) AS RN
INTO #Allergy
-- Referrals
FROM pimsmarts.dbo.referrals vwref
-- Get clinician referred to
LEFT JOIN infodb.dbo.vw_cset_prof_carers proca
ON vwref.refto_proca_refno = proca.proca_refno
-- Get source of referral
LEFT JOIN PiMSMarts.dbo.cset_sorrf sor
ON vwref.sorrf = sor.identifier
-- Outpatient Activity
LEFT JOIN InfoDB.dbo.vw_outpatients AS OUTPAT
ON vwref.refrl_refno = OUTPAT.refrl_refno
--Filter to referals to the alergy service between the specified dates
WHERE refto_local_spec IN ('3H') -- allergy service
AND vwref.create_dttm BETWEEN @startdate AND @enddate
AND cancd_dttm IS NULL


SELECT pasid, patnt_refno, refrl_refno, [Referral Date], [Days Wait to be Seen],
[Referral Clinician], [Referral Source], [Date Seen], [Appoinment Details],
[Patient Age at Appt]
FROM #Allergy WHERE RN = 1
"""
hosp_pop_sql = """SET NOCOUNT ON
SELECT [patnt_refno], [pasid],
	   DATEDIFF(hour, [pat_dob] ,GETDATE())/8766 AS [Age],
	   SEX.[description] AS [Sex],
	   ETH.[description] AS [Ethnicity],
	   [pat_pcode],
       IMD.[IndexValue] AS [IMD]
FROM [PiMSMarts].[dbo].[patients] as PAT
--sexx cset
LEFT JOIN [PiMSMarts].[dbo].[cset_sexxx] AS SEX
ON PAT.sexxx = SEX.identifier
--ethgr cset
LEFT JOIN [PiMSMarts].[dbo].[cset_ethgr] AS ETH
ON PAT.ethgr = ETH.identifier
--IMD
LEFT JOIN [PiMSMarts].[Reference].[vw_IndicesOfMultipleDeprivation2019_DecileByPostcode] AS IMD
ON PAT.pat_pcode = IMD.PostcodeFormatted
WHERE pat_dod is NULL
	  AND pat_dob > DATEADD(YEAR, -116, getdate())
"""
referals = pd.read_sql(allergy_sql, sdmart_engine)
hosp_pop = pd.read_sql(hosp_pop_sql, sdmart_engine)
sdmart_engine.dispose()

#Split IMD value into IMD1-2 and 3-10
hosp_pop['IMD'] = hosp_pop['IMD'].astype(float)
hosp_pop['IMD_split'] = np.where(hosp_pop['IMD'] <=2 , 'IMD 1-2', 'IMD 3-10')
hosp_pop.loc[hosp_pop['IMD'].isna(), 'IMD_split'] = np.nan

#Group up all ethnicities other than white british and unknown into ethnic
# minority group.
hosp_pop.loc[hosp_pop['Ethnicity'] == 'Unwilling to answer',
             'Ethnicity'] = 'Unknown'
hosp_pop.loc[~hosp_pop['Ethnicity'].isin(['White British', 'Unknown']),
             'Ethnicity'] = 'Ethnic Minority'

#Put ages into age bands
age_band_criteria = [hosp_pop['Age'] < 10,
                     (hosp_pop['Age'] >= 10) & (hosp_pop['Age'] < 20),
                     (hosp_pop['Age'] >= 20) & (hosp_pop['Age'] < 30),
                     (hosp_pop['Age'] >= 30) & (hosp_pop['Age'] < 40),
                     (hosp_pop['Age'] >= 40) & (hosp_pop['Age'] < 50),
                     (hosp_pop['Age'] >= 50) & (hosp_pop['Age'] < 60),
                     (hosp_pop['Age'] >= 60) & (hosp_pop['Age'] < 70),
                     (hosp_pop['Age'] >= 70) & (hosp_pop['Age'] < 80),
                     (hosp_pop['Age'] >= 80) & (hosp_pop['Age'] < 90),
                     (hosp_pop['Age'] >= 90) & (hosp_pop['Age'] < 100),
                     (hosp_pop['Age'] >= 100)]
age_band_labels = (['Under 10']
                    + [f'{int(i)} - {int(i+9)}' for i in np.linspace(10, 90, 9)]
                    + ['100 +'])
hosp_pop['Age Bands'] = np.select(age_band_criteria, age_band_labels)


#Add Demographic data from hosp_pop onto referrals dataset, remove allergy
#referals from hospital data
referals = referals.merge(hosp_pop, on=['pasid', 'patnt_refno'], how='left')
hosp_pop = hosp_pop.loc[~hosp_pop['pasid'].isin(referals['pasid']
                                                .drop_duplicates())].copy()
hosp_pop['type'] = 'Hospital Population'
referals['type'] = 'Allergy Referrals'

# =============================================================================
# % Functions
# =============================================================================
#Get distribution of IMD groups
def IMD_dist(df, col):
    group = (df.groupby('IMD', as_index=False)[col].count()
             .rename(columns={col:'Count'}).astype(int).sort_values(by='IMD'))
    group['Proportion'] = (group['Count']/group['Count'].sum())*100
    return group

#Get distribution of age groups
def AGE_dist(df, col):
    group = pd.DataFrame(df.groupby('Age Bands')[col].count()
                         ).rename(columns={col:'Count'})
    group['Proportion'] = (group/group.sum())*100
    missing_bands = [band for band in age_band_labels if band not in group.index]
    for band in missing_bands:
        group.loc[band] = 0
    return group.loc[age_band_labels]

#Function to plot labels on bars
def show_values_on_bars(axs,percentage = True, rounded = 2, numbers = None):
    def _show_on_single_plot(ax):
        counter = 0       
        for p in ax.patches:
            if p._height !=0:
                if percentage:
                    _x = p.get_x() + p.get_width() / 2
                    if (p.xy[1] == 0) and (p._height < 1):#p._height < 0.50:
                        _y = p.get_y() + p.get_height() + 0.01

                        if rounded == 2:
                            value = '{:.2f}'.format(p.get_height()*100)
                        elif rounded == 0:
                            value = '{:.0f}'.format(p.get_height()*100)
                        if numbers:
                            ax.text(_x, _y+0.02, value+"%\n("+str(numbers[counter])+")",
                                    ha="center", fontsize=14)
                            counter = counter + 1
                        else:
                            ax.text(_x, _y, value+"%", ha="center", fontsize=14)
                else:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height()+0.3
                    value = '{:.1f}'.format(p.get_height())
                    ax.text(_x, _y, value, ha="center", fontsize=14) 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        
#Function to show totals at top of bars
def show_totals(axs, totals):
    def _show_single_totals(ax):
        counter = 0
        for p in ax.patches:
            if p._height != 0:
                _x = p.get_x() + p.get_width() / 2
                if ((p.xy[1] > 0) and (p._height < 1)) or (p._height == 1):#p._height > 0.50:
                    _y = p.get_y() + p.get_height()-0.05
                    ax.text(_x, _y, str(totals[counter]), ha="center", fontsize=14)
                    counter = counter + 1
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_single_totals(ax)
    else:
        _show_single_totals(axs)
        
#Pie formatting
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format

#Significance bars
def label_diff(ax, i, j, text, X, Y):
    x = (X[i] + X[j]) / 2
    y = 1.07 * max(Y[i], Y[j])
    props = {'connectionstyle':'bar', 'arrowstyle':'-', 'shrinkA':20,
             'shrinkB':20,'linewidth':2}
    ylims = ax.get_ylim()[1] - ax.get_ylim()[0]
    #If its a percentage plot, don't need extra y increase
    if max(Y) == min(Y) == 1:
        ax.annotate(text, xy=(x, y*1.05), zorder=10, ha='center',
                    annotation_clip=False)
        ax.annotate('', xy=(X[i], y*0.87), xytext=(X[j], y*0.87),
                    arrowprops=props, annotation_clip=False)
    else:
        ax.annotate(text, xy=(x, y+0.2*ylims), zorder=10, ha='center')
        ax.annotate('', xy=(X[i], y), xytext=(X[j], y), arrowprops=props)        

#Test whether two proportions are different
def propHypothesisTest(p1, p2, n1, n2, alpha = 0.05):
    #Following:https://medium.com/analytics-vidhya/testing-a-difference-in-population-proportions-in-python-89d57a06254
    #p1 and p2 are the proportions of each dataset falling in the 'yes' category
    #n1 and n2 are the total number of datapoints in each dataset
    #Alpha is the signifficance threshold (10% for this 2-tailed test)
    #Null Hypothesis: Proportions equal
    #Alternative: Proportions significantly different
    #First, find the standard error
    #For this, we need the total proportion with a yes classification
    p = (n1*p1 + n2*p2)/(n1 + n2)
    se = np.sqrt(p*(1-p)*((1/n1) + (1/n2)))
    #Next, calculate the test statistic:
        #(best estimate - hypothesized estimate)/standard error
        #best estimate = p1-p2, hypothesized = 0(as p1 and p2 are equal)
    if se == 0:
        return None
    test_stat = (p1-p2)/se 
    #This gives number of standard deviations from hypothesized estimate
    #From the test statistic, get the p-value
    pvalue = 2 * dist.norm.cdf(-np.abs(test_stat)) # Multiplied by two indicates a two tailed testing.
    return pvalue

#function to get list of counts to account for missing values
def counts_list(counts, options):
    c1 = (counts[options[0]] if options[0] in counts.index else 0)
    c2 = (counts[options[1]] if options[1] in counts.index else 0)
    return [c1, c2]


# =============================================================================
# =============================================================================
#   #Inequalities between hosp pop and those referred to the allergy service
# =============================================================================
# =============================================================================

# =============================================================================
									#IMD
# =============================================================================
#Get the proportion of each population that fall within each IMD
hosp_pop_IMD = IMD_dist(hosp_pop, 'pasid')
referals_IMD = IMD_dist(referals, 'pasid')

						#####Statistical Test#####
#Do a statistical test to see if the differences are significant.
pvalue = mannwhitneyu(referals['IMD'], hosp_pop['IMD'], alternative='greater',
                      nan_policy='omit')[1]

							#####Plot#####
# Figure setup
plt.figure(figsize=(20, 20))
ys = range(10)[::-1]
height = 0.8
base = 0
# Draw bars for Community
for y, value in zip(ys, hosp_pop_IMD['Proportion'].values):
    plt.broken_barh([(base, -np.abs(base-value))], (y - height/2,height),
                    facecolors=['#0d47a1','#0d47a1'], label='Hospital Population')
# Draw bars for Allergy referals
for y, value2 in zip(ys, referals_IMD['Proportion'].values):
    plt.broken_barh([(base, np.abs(base-value2))], (y - height/2, height),
                    facecolors=['#e2711d','#e2711d'], label='Allergy Referals')
#If statistically significant difference, add text to say this.
if pvalue < 0.05:
    plt.text(-20, 3, f'Referals to allergy service\nhave a higher IMD than\n the general Hospital population \np-value={pvalue:.2e}',
        fontsize=16, horizontalalignment='center', verticalalignment='center')
#Plot difference in proportions
plt.plot((referals_IMD['Proportion'] - hosp_pop_IMD['Proportion']).values,
          ys, 'ok', markersize=20, label='Difference')
# Modify the graph
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=16)
plt.yticks(ys, referals_IMD['IMD'].values.tolist())
plt.xticks(np.linspace(-30, 30, 13),
           [abs(int(i)) for i in np.linspace(-30, 30, 13)])
plt.xlabel('% of population in each IMD', fontsize=20)
plt.ylabel('IMD', fontsize=20)
plt.grid(linewidth=0.1, color='black')
plt.title('Proportion of Allergy referals by IMD compared to the general Hospital population', fontsize=20)
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig('Plots/IMD referals.png', bbox_inches='tight')
plt.close()
# =============================================================================
									#Age
# =============================================================================
def AGE_dist(df, col):
    group = pd.DataFrame(df.groupby('Age Bands')[col].count()
                         ).rename(columns={col:'Count'})
    group['Proportion'] = (group/group.sum())*100
    missing_bands = [band for band in age_band_labels if band not in group.index]
    for band in missing_bands:
        group.loc[band] = 0
    return group.loc[age_band_labels]

hosp_pop_age = AGE_dist(hosp_pop, 'pasid')
referals_age = AGE_dist(referals, 'pasid')

					  #####Statistical Test#####
#Do a statistical test to see if the differences are significant.
pvalue = mannwhitneyu(referals['Age'], hosp_pop['Age'], alternative='less',
                      nan_policy='omit')[1]

							#####Plot#####
# Figure setup
plt.figure(figsize=(20, 20))
ys = range(len(age_band_labels))
height = 0.8
base = 0
# Draw bars for Community
for y, value in zip(ys, hosp_pop_age['Proportion'].values[::-1]):
    plt.broken_barh([(base, -np.abs(base-value))], (y - height/2,height),
                    facecolors=['#0d47a1','#0d47a1'], label='Hospital Population')
# Draw bars for Allergy referals
for y, value2 in zip(ys, referals_age['Proportion'].values[::-1]):
    plt.broken_barh([(base, np.abs(base-value2))], (y - height/2, height),
                    facecolors=['#e2711d','#e2711d'], label='Allergy Referals')
#If statistically significant difference, add text to say this.
if pvalue < 0.05:
    plt.text(-20, 3, f'Referals to allergy service\nhave a lower age than\n the general Hospital population \np-value={pvalue:.2e}',
        fontsize=16, horizontalalignment='center', verticalalignment='center')
#Plot difference in proportions
plt.plot((referals_age['Proportion'] - hosp_pop_age['Proportion']).values[::-1],
          ys, 'ok', markersize=20, label='Difference')
# Modify the graph
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=16)
plt.yticks(ys, age_band_labels[::-1], fontsize=14)
plt.xticks(np.linspace(-30, 30, 13),
           [abs(int(i)) for i in np.linspace(-30, 30, 13)], fontsize=14)
plt.xlabel('% of population in each Age Band', fontsize=20)
plt.ylabel('Age Band', fontsize=20)
plt.grid(linewidth=0.1, color='black')
plt.title('Proportion of Allergy referals by Age compared to the general Hospital population', fontsize=20)
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig('Plots/Age referals.png', bbox_inches='tight')
plt.close()

# =============================================================================
									#Ethnicity
# =============================================================================
hosp_eth = hosp_pop.loc[hosp_pop['Ethnicity'] != 'Unknown', ['Ethnicity', 'type']].copy()
referal_eth = referals.loc[referals['Ethnicity'] != 'Unknown', ['Ethnicity', 'type']].copy()
hosp_eth_counts = hosp_eth['Ethnicity'].value_counts()
referal_eth_counts = referal_eth['Ethnicity'].value_counts()

pvalue = propHypothesisTest(hosp_eth_counts['Ethnic Minority']/hosp_eth_counts.sum(),
                            referal_eth_counts['Ethnic Minority']/referal_eth_counts.sum(),
                            hosp_eth_counts['Ethnic Minority'],
                            referal_eth_counts['Ethnic Minority'],
                            alpha=0.05)
text = 'a' if pvalue < 0.05 else 'no'

#Plot a filled bar
fig, ax = plt.subplots(1,1, figsize=(15,15))
hue_order = ['White British', 'Ethnic Minority']
data_sets = ['Hospital Population', 'Allergy Referrals']
sns.histplot(data=pd.concat([hosp_eth, referal_eth]), x='type', hue='Ethnicity',
             multiple='fill', shrink=0.6, hue_order=hue_order,
             palette=['lightskyblue','royalblue'],
             alpha=1, ax=ax)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('', fontsize=20)
ax.set_ylabel('Percentage of Patients', fontsize=20)
ax.set_title('Allergy Referals vs Hospital Population on Ethinicity', fontsize=20)
legend = ax.get_legend()
# #Get seaborn legend
handles = legend.legend_handles
ax.legend(handles,
               ['White British', 'Ethnic Minority'],
               bbox_to_anchor=(1,1), fontsize=14)
#Make a list of the numbers to include under the percentages
numbers = [hosp_eth_counts['Ethnic Minority'], referal_eth_counts['Ethnic Minority']]
totals = [hosp_eth_counts.sum(), referal_eth_counts.sum()]
show_values_on_bars(ax, numbers = [i for i in numbers if i!=0])
show_totals(ax, totals)
#add text box
fig.text(.46, .4,
         tw.fill(f"There is {text} significant difference between the "\
                 "proportion of ethnic minority patients referred to the allergy"\
                 "service and those in the general hospital population.", 28),
         ha='center', clip_on=False, fontsize=14,
         bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='black'))
plt.tight_layout()
plt.savefig('plots/Ethnicity referals.png', bbox_inches='tight')
plt.close()

# =============================================================================
									#Sex
# =============================================================================
hosp_sex = hosp_pop.loc[hosp_pop['Sex'].isin(['Male', 'Female']), ['Sex', 'type']].copy()
referal_sex = referals.loc[referals['Sex'].isin(['Male', 'Female']), ['Sex', 'type']].copy()
hosp_sex_counts = hosp_sex['Sex'].value_counts()
referal_sex_counts = referal_sex['Sex'].value_counts()

pvalue = propHypothesisTest(hosp_sex_counts['Female']/hosp_sex_counts.sum(),
                            referal_sex_counts['Female']/referal_sex_counts.sum(),
                            hosp_sex_counts['Female'],
                            referal_sex_counts['Female'],
                            alpha=0.05)
text = 'a' if pvalue < 0.05 else 'no'

#Plot a filled bar
fig, ax = plt.subplots(1,1, figsize=(15,15))
hue_order = ['Male', 'Female']
data_sets = ['Hospital Population', 'Allergy Referrals']
sns.histplot(data=pd.concat([hosp_sex, referal_sex]), x='type', hue='Sex',
             multiple='fill', shrink=0.6, hue_order=hue_order,
             palette=['lightskyblue','royalblue'],
             alpha=1, ax=ax)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('', fontsize=20)
ax.set_ylabel('Percentage of Patients', fontsize=20)
ax.set_title('Allergy Referals vs Hospital Population on Sex', fontsize=20)
legend = ax.get_legend()
# #Get seaborn legend
handles = legend.legend_handles
ax.legend(handles, ['Male', 'Female'], bbox_to_anchor=(1,1), fontsize=14)
#Make a list of the numbers to include under the percentages
numbers = [hosp_sex_counts['Female'], referal_sex_counts['Female']]
totals = [hosp_sex_counts.sum(), referal_sex_counts.sum()]
show_values_on_bars(ax, numbers = [i for i in numbers if i!=0])
show_totals(ax, totals)
#add text box
fig.text(.485, .4,
         tw.fill(f"There is {text} significant difference between the "\
                 "proportion of female patients referred to the allergy"\
                 "service and those in the general hospital population.", 28),
         ha='center', clip_on=False, fontsize=14,
         bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='black'))
plt.tight_layout()
plt.savefig('plots/Sex referals.png', bbox_inches='tight')
plt.close()














# =============================================================================
# =============================================================================
#   #Inequalitites in those who have and have not been seen from the allgergy service
# =============================================================================
# =============================================================================
seen = referals.loc[~referals['Date Seen'].isna()].copy()
seen['type'] = 'Seen'
wait = referals.loc[referals['Date Seen'].isna()].copy()
wait['type'] = 'Waiting'

# =============================================================================
									#IMD
# =============================================================================
#Get the proportion of each population that fall within each IMD
seen_IMD = IMD_dist(seen, 'pasid')
wait_IMD = IMD_dist(wait, 'pasid')

						#####Statistical Test#####
#Do a statistical test to see if the differences are significant.
pvalue = mannwhitneyu(seen['IMD'], wait['IMD'], alternative='greater',
                      nan_policy='omit')[1]
text = ('Those seen by the Allergy Servive have a higher IMD than those still waiting'
        if pvalue < 0.05 else 'No significant statistical difference in IMD')

							#####Plot#####
# Figure setup
plt.figure(figsize=(20, 20))
ys = range(10)[::-1]
height = 0.8
base = 0
# Draw bars for Community
for y, value in zip(ys, wait_IMD['Proportion'].values):
    plt.broken_barh([(base, -np.abs(base-value))], (y - height/2,height),
                    facecolors=['#0d47a1','#0d47a1'], label='Waiting')
# Draw bars for Allergy seen
for y, value2 in zip(ys, seen_IMD['Proportion'].values):
    plt.broken_barh([(base, np.abs(base-value2))], (y - height/2, height),
                    facecolors=['#e2711d','#e2711d'], label='Seen')
#If statistically significant difference, add text to say this.
plt.text(-20, 3, tw.fill(text, 28), ha='center', clip_on=False, fontsize=16)
#Plot difference in proportions
plt.plot((seen_IMD['Proportion'] - wait_IMD['Proportion']).values,
          ys, 'ok', markersize=20, label='Difference')
# Modify the graph
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=16)
plt.yticks(ys, seen_IMD['IMD'].values.tolist())
plt.xticks(np.linspace(-30, 30, 13),
           [abs(int(i)) for i in np.linspace(-30, 30, 13)])
plt.xlabel('% of population in each IMD', fontsize=20)
plt.ylabel('IMD', fontsize=20)
plt.grid(linewidth=0.1, color='black')
plt.title('Proportion of those seen by Allergy referals by IMD compared to those waiting to be seen', fontsize=20)
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig('Plots/IMD seen.png', bbox_inches='tight')
plt.close()
# =============================================================================
									#Age
# =============================================================================
wait_age = AGE_dist(wait, 'pasid')
seen_age = AGE_dist(seen, 'pasid')

					  #####Statistical Test#####
#Do a statistical test to see if the differences are significant.
pvalue = mannwhitneyu(seen['Age'], wait['Age'], alternative='greater',
                      nan_policy='omit')[1]
text = ('Those seen by the Allergy Service have a higher age than those still waiting'
        if pvalue < 0.05 else 'No significan statistical difference in age')

							#####Plot#####
# Figure setup
plt.figure(figsize=(20, 20))
ys = range(len(age_band_labels))
height = 0.8
base = 0
# Draw bars for Community
for y, value in zip(ys, wait_age['Proportion'].values[::-1]):
    plt.broken_barh([(base, -np.abs(base-value))], (y - height/2,height),
                    facecolors=['#0d47a1','#0d47a1'], label='Waiting')
# Draw bars for Allergy referals
for y, value2 in zip(ys, seen_age['Proportion'].values[::-1]):
    plt.broken_barh([(base, np.abs(base-value2))], (y - height/2, height),
                    facecolors=['#e2711d','#e2711d'], label='Seen')
#If statistically significant difference, add text to say this.
plt.text(-20, 3, tw.fill(text, 28), ha='center', clip_on=False, fontsize=16)
#Plot difference in proportions
plt.plot((seen_age['Proportion'] - wait_age['Proportion']).values[::-1],
          ys, 'ok', markersize=20, label='Difference')
# Modify the graph
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=16)
plt.yticks(ys, age_band_labels[::-1], fontsize=14)
plt.xticks(np.linspace(-30, 30, 13),
           [abs(int(i)) for i in np.linspace(-30, 30, 13)], fontsize=14)
plt.xlabel('% of population in each Age Band', fontsize=20)
plt.ylabel('Age Band', fontsize=20)
plt.grid(linewidth=0.1, color='black')
plt.title('Proportion of those seen by Allergy referals by age compared to those waiting to be seen', fontsize=20)
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig('Plots/Age seen.png', bbox_inches='tight')
plt.close()

# =============================================================================
									#Ethnicity
# =============================================================================
wait_eth = wait.loc[wait['Ethnicity'] != 'Unknown', ['Ethnicity', 'type']].copy()
seen_eth = seen.loc[seen['Ethnicity'] != 'Unknown', ['Ethnicity', 'type']].copy()
wait_eth_counts = wait_eth['Ethnicity'].value_counts()
seen_eth_counts = seen_eth['Ethnicity'].value_counts()

pvalue = propHypothesisTest(wait_eth_counts['Ethnic Minority']/wait_eth_counts.sum(),
                            seen_eth_counts['Ethnic Minority']/seen_eth_counts.sum(),
                            wait_eth_counts['Ethnic Minority'],
                            seen_eth_counts['Ethnic Minority'],
                            alpha=0.05)
text = 'a' if pvalue < 0.05 else 'no'

#Plot a filled bar
fig, ax = plt.subplots(1,1, figsize=(15,15))
hue_order = ['White British', 'Ethnic Minority']
sns.histplot(data=pd.concat([wait_eth, seen_eth]), x='type', hue='Ethnicity',
             multiple='fill', shrink=0.6, hue_order=hue_order,
             palette=['lightskyblue','royalblue'],
             alpha=1, ax=ax)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('', fontsize=20)
ax.set_ylabel('Percentage of Patients', fontsize=20)
ax.set_title('Allergy Seen vs Allergy Waiting on Ethinicity', fontsize=20)
legend = ax.get_legend()
# #Get seaborn legend
handles = legend.legend_handles
ax.legend(handles, ['White British', 'Ethnic Minority'], bbox_to_anchor=(1,1),
          fontsize=14)
#Make a list of the numbers to include under the percentages
numbers = [wait_eth_counts['Ethnic Minority'], seen_eth_counts['Ethnic Minority']]
totals = [wait_eth_counts.sum(), seen_eth_counts.sum()]
show_values_on_bars(ax, numbers = [i for i in numbers if i!=0])
show_totals(ax, totals)
#add text box
fig.text(.46, .4,
         tw.fill(f"There is {text} significant difference between the "\
                 "proportion of ethnic minority seen by the allergy "\
                 "service and those still on the wait list.", 28),
         ha='center', clip_on=False, fontsize=14,
         bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='black'))
plt.tight_layout()
plt.savefig('plots/Ethnicity seen.png', bbox_inches='tight')
plt.close()

# =============================================================================
									#Sex
# =============================================================================
wait_sex = wait.loc[wait['Sex'].isin(['Male', 'Female']), ['Sex', 'type']].copy()
seen_sex = seen.loc[seen['Sex'].isin(['Male', 'Female']), ['Sex', 'type']].copy()
wait_sex_counts = wait_sex['Sex'].value_counts()
seen_sex_counts = seen_sex['Sex'].value_counts()

pvalue = propHypothesisTest(wait_sex_counts['Female']/wait_sex_counts.sum(),
                            seen_sex_counts['Female']/seen_sex_counts.sum(),
                            wait_sex_counts['Female'],
                            seen_sex_counts['Female'],
                            alpha=0.05)
text = 'a' if pvalue < 0.05 else 'no'

#Plot a filled bar
fig, ax = plt.subplots(1,1, figsize=(15,15))
hue_order = ['Male', 'Female']
sns.histplot(data=pd.concat([wait_sex, seen_sex]), x='type', hue='Sex',
             multiple='fill', shrink=0.6, hue_order=hue_order,
             palette=['lightskyblue','royalblue'],
             alpha=1, ax=ax)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('', fontsize=20)
ax.set_ylabel('Percentage of Patients', fontsize=20)
ax.set_title('Allergy Seen vs Allergy Waiting on Sex', fontsize=20)
legend = ax.get_legend()
# #Get seaborn legend
handles = legend.legend_handles
ax.legend(handles, ['Male', 'Female'], bbox_to_anchor=(1,1), fontsize=14)
#Make a list of the numbers to include under the percentages
numbers = [wait_sex_counts['Female'], seen_sex_counts['Female']]
totals = [wait_sex_counts.sum(), seen_sex_counts.sum()]
show_values_on_bars(ax, numbers = [i for i in numbers if i!=0])
show_totals(ax, totals)
#add text box
fig.text(.485, .4,
         tw.fill(f"There is {text} significant difference between the "\
                 "proportion of female seen by the allergy "\
                 "service and those still waiting.", 28),
         ha='center', clip_on=False, fontsize=14,
         bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='black'))
plt.tight_layout()
plt.savefig('plots/Sex seen.png', bbox_inches='tight')
plt.close()








# =============================================================================
# =============================================================================
#   #Inequalities in length of wait for those who have been seen by the allergy service
# =============================================================================
# =============================================================================
bins = [i*20 for i in range(0, 25)]

# =============================================================================
									#IMD
# =============================================================================
seen_IMD_12 = seen.loc[seen['IMD_split'] == 'IMD 1-2'].copy()
seen_IMD_310 = seen.loc[seen['IMD_split'] == 'IMD 3-10'].copy()

						#####Statistical Test#####
#Do a statistical test to see if the differences are significant.
pvalue = mannwhitneyu(seen_IMD_12['Days Wait to be Seen'],
                      seen_IMD_310['Days Wait to be Seen'],
                      alternative='greater',
                      nan_policy='omit')[1]
text = ('Those in lower IMDs have a longer wait to be seen than those in higher IMDs'
        if pvalue < 0.05 else 'No siginificant statistical difference in time to be seen by IMD')

							#####Plot#####
count_12 = np.histogram(seen_IMD_12['Days Wait to be Seen'], bins=bins)[0]
count_310 = np.histogram(seen_IMD_310['Days Wait to be Seen'], bins=bins)[0]
IMD_wait = pd.DataFrame({'Bins':bins[:-1],
                         'count 1-2':count_12,
                         'count 3-10':count_310})
IMD_wait[['proportion 1-2', 'proportion 3-10']] = (
    									  IMD_wait[['count 1-2', 'count 3-10']]
                                        / IMD_wait[['count 1-2', 'count 3-10']]
                                        .sum())*100
# Figure setup
plt.figure(figsize=(20, 20))
height = bins[1] - bins[0]
base = 0
# Draw bars for Community
for y, value in zip(bins, IMD_wait['proportion 1-2'].values):
    plt.broken_barh([(base, -np.abs(base-value))], (y, height),
                    facecolors=['#0d47a1','#0d47a1'], label='IMD 1-2')
# Draw bars for Allergy seen
for y, value2 in zip(bins, IMD_wait['proportion 3-10'].values):
    plt.broken_barh([(base, np.abs(base-value2))], (y, height),
                    facecolors=['#e2711d','#e2711d'], label='IMD 3-10')
#If statistically significant difference, add text to say this.
plt.text(-15, 5, tw.fill(text, 28), ha='center', clip_on=False, fontsize=16)
#Plot difference in proportions
plt.plot((IMD_wait['proportion 3-10'] -IMD_wait['proportion 1-2']).values,
          [bin+10 for bin in bins[:-1]], '-k', label='Difference')
# Modify the graph
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=16)
plt.yticks(bins, bins)
plt.xticks(np.linspace(-25, 25, 13),
           [abs(int(i)) for i in np.linspace(-30, 30, 13)])
plt.xlabel('% of population', fontsize=20)
plt.ylabel('Days Wait to be Seen', fontsize=20)
plt.grid(linewidth=0.1, color='black')
plt.title('Number of Days Wait to be Seen by IMD', fontsize=20)
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig('Plots/IMD wait.png', bbox_inches='tight')
plt.close()

# =============================================================================
									#Ethnicity
# =============================================================================
seen_ethm = seen.loc[seen['Ethnicity'] == 'Ethnic Minority'].copy()
seen_wb = seen.loc[seen['Ethnicity'] != 'Ethnic Minority'].copy()

						#####Statistical Test#####
#Do a statistical test to see if the differences are significant.
pvalue = mannwhitneyu(seen_ethm['Days Wait to be Seen'],
                      seen_wb['Days Wait to be Seen'],
                      alternative='greater',
                      nan_policy='omit')[1]
text = ('Ethnic Minority patients have a longer wait to be seen than White British/Unkown patients'
        if pvalue < 0.05 else 'No siginificant statistical difference in time to be seen by Ethnicity')

							#####Plot#####
count_ethm = np.histogram(seen_ethm['Days Wait to be Seen'], bins=bins)[0]
count_wb = np.histogram(seen_wb['Days Wait to be Seen'], bins=bins)[0]
eth_wait = pd.DataFrame({'Bins':bins[:-1],
                         'count Ethnic Minority':count_ethm,
                         'count White British/Unkown':count_wb})
eth_wait[['proportion Ethnic Minority',
          'proportion White British/Unkown']] = (eth_wait[
              									['count Ethnic Minority',
                        						 'count White British/Unkown']]
                                                / eth_wait[
                                                ['count Ethnic Minority',
                                                 'count White British/Unkown']]
                                                 .sum())*100
# Figure setup
plt.figure(figsize=(20, 20))
height = bins[1] - bins[0]
base = 0
# Draw bars for Community
for y, value in zip(bins, eth_wait['proportion Ethnic Minority'].values):
    plt.broken_barh([(base, -np.abs(base-value))], (y, height),
                    facecolors=['#0d47a1','#0d47a1'], label='Ethnic Minority')
# Draw bars for Allergy seen
for y, value2 in zip(bins, eth_wait['proportion White British/Unkown'].values):
    plt.broken_barh([(base, np.abs(base-value2))], (y, height),
                    facecolors=['#e2711d','#e2711d'], label='White British/Unkown')
#If statistically significant difference, add text to say this.
plt.text(-15, 5, tw.fill(text, 28), ha='center', clip_on=False, fontsize=16)
#Plot difference in proportions
plt.plot((eth_wait['proportion White British/Unkown'] - eth_wait['proportion Ethnic Minority']).values,
          [bin+10 for bin in bins[:-1]], '-k', label='Difference')
# Modify the graph
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=16)
plt.yticks(bins, bins)
plt.xticks(np.linspace(-25, 25, 13),
           [abs(int(i)) for i in np.linspace(-30, 30, 13)])
plt.xlabel('% of population', fontsize=20)
plt.ylabel('Days Wait to be Seen', fontsize=20)
plt.grid(linewidth=0.1, color='black')
plt.title('Number of Days Wait to be Seen by Ethnicity', fontsize=20)
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig('Plots/Ethnicity wait.png', bbox_inches='tight')
plt.close()

# =============================================================================
									#Gender
# =============================================================================
seen_fem = seen.loc[seen['Sex'] == 'Female'].copy()
seen_male = seen.loc[seen['Sex'] != 'Male'].copy()

						#####Statistical Test#####
#Do a statistical test to see if the differences are significant.
pvalue = mannwhitneyu(seen_fem['Days Wait to be Seen'],
                      seen_male['Days Wait to be Seen'],
                      alternative='greater',
                      nan_policy='omit')[1]
text = ('Female patients have a longer wait to be seen than Male patients'
        if pvalue < 0.05 else 'No siginificant statistical difference in time to be seen by Sex')

							#####Plot#####
count_fem = np.histogram(seen_fem['Days Wait to be Seen'], bins=bins)[0]
count_male = np.histogram(seen_male['Days Wait to be Seen'], bins=bins)[0]
sex_wait = pd.DataFrame({'Bins':bins[:-1],
                         'count Female':count_fem,
                         'count Male':count_male})
sex_wait[['proportion Female',
          'proportion Male']] = (sex_wait[['count Female', 'count Male']]
                                 / sex_wait[['count Female', 'count Male']]
                                 .sum())*100
# Figure setup
plt.figure(figsize=(20, 20))
height = bins[1] - bins[0]
base = 0
# Draw bars for Community
for y, value in zip(bins, sex_wait['proportion Female'].values):
    plt.broken_barh([(base, -np.abs(base-value))], (y, height),
                    facecolors=['#0d47a1','#0d47a1'], label='Female')
# Draw bars for Allergy seen
for y, value2 in zip(bins, sex_wait['proportion Male'].values):
    plt.broken_barh([(base, np.abs(base-value2))], (y, height),
                    facecolors=['#e2711d','#e2711d'], label='Male')
#If statistically significant difference, add text to say this.
plt.text(-15, 5, tw.fill(text, 28), ha='center', clip_on=False, fontsize=16)
#Plot difference in proportions
plt.plot((sex_wait['proportion Male'] - sex_wait['proportion Female']).values,
          [bin+10 for bin in bins[:-1]], '-k', label='Difference')
# Modify the graph
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=16)
plt.yticks(bins, bins)
plt.xticks(np.linspace(-25, 25, 13),
           [abs(int(i)) for i in np.linspace(-30, 30, 13)])
plt.xlabel('% of population', fontsize=20)
plt.ylabel('Days Wait to be Seen', fontsize=20)
plt.grid(linewidth=0.1, color='black')
plt.title('Number of Days Wait to be Seen by Sex', fontsize=20)
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig('Plots/Sex wait.png', bbox_inches='tight')
plt.close()

# =============================================================================
									#Age
# =============================================================================
seen_age = seen[['Age', 'Days Wait to be Seen']].dropna()

pvalue = pearsonr(seen_age['Age'], seen_age['Days Wait to be Seen'])[1]
text = ('There is a statistically significant difference in how long patients wait to be seen by age'
        if pvalue < 0.05 else 'No statistically siginificant difference in time to be seen by Age')

# Figure setup
plt.figure(figsize=(20, 20))
ax = sns.regplot(seen_age, x='Age', y='Days Wait to be Seen', line_kws=dict(color="r"))
#If statistically significant difference, add text to say this.
plt.text(75, 300, tw.fill(text, 28), ha='center', clip_on=False, fontsize=16)
plt.title('Number of Days Wait to be Seen by Age', fontsize=20)
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig('Plots/Age wait.png', bbox_inches='tight')
plt.close()


