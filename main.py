import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import IPython.display as ipd
from tabulate import tabulate
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier



Seeds = pd.read_csv('Dataformachinelearning/MNCAATourneySeeds.csv')
Conferences = pd.read_csv('Dataformachinelearning/MTeamConferences.csv')
RegularDetail = pd.read_csv('Dataformachinelearning/MRegularSeasonDetailedResults.csv')
TourneyCompact = pd.read_csv('Dataformachinelearning/MNCAATourneyCompactResults.csv')

winteams = pd.DataFrame()
loseteams = pd.DataFrame()


columns = ['Season' , 'TeamID', 'Points', 'OppPoints', 'Loc', 'NumOT',
 'FGM' ,'FGA' ,'FGM3' ,'FGA3', 'FTM', 'FTA', 'OR', 'DR' ,'Ast', 'TO',
 'Stl' ,'Blk' ,'PF', 'OppFGM', 'OppFGA', 'OppFGM3' ,'OppFGA3' ,'OppFTM' ,'OppFTA', 'OppOR',
 'OppDR' ,'OppAst' ,'OppTO', 'OppStl', 'OppBlk', 'OppPF']



winteams[columns] = RegularDetail[['Season', 'WTeamID', 'WScore', 'LScore', 'WLoc', 'NumOT',
 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM' ,'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3' ,'LFTM', 'LFTA', 'LOR',
 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk' ,'LPF']]
winteams['Wins'] = 1
winteams['Losses'] = 0
loseteams[columns] = RegularDetail[['Season', 'LTeamID', 'LScore', 'WScore', 'WLoc', 'NumOT',
 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM' ,'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3' ,'WFTM', 'WFTA', 'WOR',
 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk' ,'WPF']]
loseteams['Wins'] = 0
loseteams['Losses'] = 1

def change_loc(loc):
    if loc=='H':
        return 'A'
    elif loc=='A':
        return 'H'
    else:
        return 'N'

loseteams['Loc'] = loseteams['Loc'].apply(change_loc)


winloseteams = pd.concat([winteams,loseteams])

combinedteams = winloseteams.groupby(['Season', 'TeamID']).sum()
combinedteams['NumGames'] = combinedteams['Wins']+combinedteams['Losses']
#ipd.display(combinedteams)

regularseasoninput = pd.DataFrame()
#basic team metrics
regularseasoninput['WinRatio'] = combinedteams['Wins']/combinedteams['NumGames']
regularseasoninput['PointsPerGame'] = combinedteams['Points']/combinedteams['NumGames']
regularseasoninput['PointsAllowed'] = combinedteams['OppPoints']/combinedteams['NumGames']
regularseasoninput['PointsRatio'] = combinedteams['Points']/combinedteams['OppPoints']
regularseasoninput['OTsPerGame']= combinedteams['NumOT']/combinedteams['NumGames']
#Field goal all
regularseasoninput['FGPerGame'] = combinedteams['FGM']/combinedteams['NumGames']
regularseasoninput['FGRatio'] = combinedteams['FGM']/combinedteams['FGA']
regularseasoninput['FGAllowedPerGame'] = combinedteams['OppFGM']/combinedteams['NumGames']
#Fieldgoal3
regularseasoninput['FG3PerGame'] = combinedteams['FGM3']/combinedteams['NumGames']
regularseasoninput['FGR3atio'] = combinedteams['FGM3']/combinedteams['FGA3']
regularseasoninput['FGA3llowedPerGame'] = combinedteams['OppFGM3']/combinedteams['NumGames']
#Free throw metrics
regularseasoninput['FTPerGame'] = combinedteams['FTM']/combinedteams['NumGames']
regularseasoninput['FTRatio'] = combinedteams['FTM']/combinedteams['FTA']
regularseasoninput['FTAllowedPerGame'] = combinedteams['OppFTM']/combinedteams['NumGames']
#ipd.display(regularseasoninput)

#rebounds
regularseasoninput['ORRatio']=combinedteams['OR']/(combinedteams['OR']+combinedteams['OppDR'])
regularseasoninput['DRRatio']=combinedteams['DR']/(combinedteams['DR']+combinedteams['OppOR'])
regularseasoninput['AstPerGame'] = combinedteams['Ast']/combinedteams['NumGames']

regularseasoninput['TOPerGame'] = combinedteams['TO']/combinedteams['NumGames']
regularseasoninput['StlPerGame'] = combinedteams['Stl']/combinedteams['NumGames']
regularseasoninput['BlkPerGame'] = combinedteams['Blk']/combinedteams['NumGames']
regularseasoninput['PFPerGame'] = combinedteams['PF']/combinedteams['NumGames']

#ipd.display_markdown(Seeds)

#ipd.display(regularseasoninput.isna().sum())

seed_dict = Seeds.set_index(['Season','TeamID'])

ipd.display(seed_dict.columns.values)

TourneyInput = pd.DataFrame()

winIDs = TourneyCompact['WTeamID']
loseIDs = TourneyCompact['LTeamID']
season = TourneyCompact['Season']

winners = pd.DataFrame()
winners[['Season','Team1','Team2']] = TourneyCompact[['Season','WTeamID','LTeamID']]
winners['Result']=1

losers = pd.DataFrame()
losers[['Season','Team1','Team2']] = TourneyCompact[['Season','LTeamID','WTeamID']]
losers['Result']=0

TourneyInput = pd.concat([winners,losers])
TourneyInput = TourneyInput[TourneyInput['Season']>=2003].reset_index(drop=True)
#ipd.display(TourneyInput)
team1seeds = []
team2seeds = []

for x in range(len(TourneyInput)):
    idx = (TourneyInput['Season'][x],TourneyInput['Team1'][x])
    seed = seed_dict.loc[idx].values[0]
    if len(seed) == 4:
        seed = int(seed[1:-1])
    else:
        seed = int(seed[1:])
    team1seeds.append(seed)
    idx = (TourneyInput['Season'][x], TourneyInput['Team2'][x])
    seed = seed_dict.loc[idx].values[0]
    if len(seed) == 4:
        seed = int(seed[1:-1])
    else:
        seed = int(seed[1:])
    team2seeds.append(seed)
TourneyInput['Team1Seed'] = team1seeds
TourneyInput['Team2Seed'] = team2seeds

#ipd.display(TourneyInput)
outscores = []

for x in range(len(TourneyInput)):
    idx = (TourneyInput['Season'][x], TourneyInput['Team1'][x])
    team1score = regularseasoninput.loc[idx]
    team1score['Seed'] = TourneyInput['Team1Seed'][x]

    idx = (TourneyInput['Season'][x], TourneyInput['Team2'][x])
    team2score = regularseasoninput.loc[idx]
    team2score['Seed'] = TourneyInput['Team2Seed'][x]

    outscore = team1score-team2score
    outscore['Result'] = TourneyInput['Result'][x]
    outscores.append(outscore)

outscores = pd.DataFrame(outscores)

#ipd.display(outscores)
corrs = round(outscores.corr(), 2)
#ipd.display(np.abs(corrs['Result']))
'''
plt.figure()
sns.heatmap(corrs,cmap="Greys")
plt.show()'''

x = outscores[outscores.columns[:-1]].values
y = outscores['Result'].values

np.random.seed(1)
idx = np.random.permutation(len(x))
train_idx = idx[:int(-0.2*len(x))]
test_idx = idx[int(-0.2*len(x)):]

x_train = x[train_idx]
x_test = x[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]


mins = x_train.min(axis=0)
maxs =x_train.max(axis=0)

x_train = (x_train-mins)/(maxs-mins)
x_test = (x_test-mins)/(maxs-mins)

#print(x_train.shape, x_test.shape,y_train.shape, y_test.shape)

model = RandomForestClassifier(random_state=1)
model = model.fit(x_train,y_train)
model.score(x_test, y_test)
print(model.score(x_test, y_test))

