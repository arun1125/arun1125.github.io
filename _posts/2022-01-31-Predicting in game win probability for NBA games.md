# Predicting in game win probabilities of NBA games



## Why?, How?, what else has been done?

Before I moved to Canada I never really watched basketball live, mostly due to the time difference and me liking my sleep but when I did get the oppourtunity to move here in March/April 2019 I witnessed the Toronto Raptors make their championship run and ever since then i've been hooked. 

Naturally as a Data Scientist and someone who studied Maths at university I was drawn to the numbers side of things with basketball. Advanced stats, different ways of measuring impact, elo scores and other people's efforts working with basketball data showed me the other half of the sport. 

Seeing other peoples efforts really highlighted the amount of data that can be collected and analysed so I decided to do my own project with NBA data. Python has an excellent [API wrapper](https://github.com/swar/nba_api) around stats.nba.com with access to things I didn't even know were tracked. 

An interesting part of the api is that it provided live play by play data and I became interested in how nba games were predicted live.

## Reserach / Review

Looking around at what had been done I found a few blogs/articles/papers mostly related to other sports but some around basketball (NCAA and NBA).

- [Bayesian approach to predicting football(not soccer) games](https://dtai.cs.kuleuven.be/sports/blog/a-bayesian-approach-to-in-game-win-probability) [1]

- [Brian Burkes NFL forecasting](http://wagesofwins.com/2009/03/05/modeling-win-probability-for-a-college-basketball-game-a-guest-post-from-brian-burke/) [2]

- [py-ball](https://github.com/basketballrelativity/py_ball) [3]

- [inpredictable](http://stats.inpredictable.com/nba/wpBox_live.php) [4]


Reading the [1] definitely cleared up that predicting football was a lot harder since a lot of games end in draws and there is a lot of infrequent scoring.

[2] & [3] gave me the first step in order to build a model to predict games. The approach that was used here is to split the game into n-second intervals and then build a series of logisitc regression models, one for each interval. 

[4] gave me a sense of what other blogs were doing and something to compare my graphs too

# Data Gathering

As mentioned before, python has a GREAT wrapper for the stats.nba.com api again linked [here](https://github.com/swar/nba_api), which is worth checking out in your own time just to see the volume of data available to play with. But I wrote a simple script to collect all the playbyplay data for the last ~7 odd years. 
The problem is rate limits! 

```python
#script to get all live playbyplay data
from nba_api.stats.endpoints import leaguegamelog, playbyplayv2
import pandas as pd
```

```python
seasons = [f'20{x}-{x+1}' for x in range(13,22)] #can put whatever range you want

league_game_logs = []
for season in seasons:
    game_log = leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]
    league_game_logs.append(game_log)
```

```python
league_game_log = pd.concat(league_game_logs)
```

#### Filtering for home games

The reason i'm filtering for the home games below is because there can only be two outcomes in basketball, win or lose and so if you find the probability that the home team wins then you're done. The game log api wrapper returns game logs from both teams perspectives. Matchups with '@' in them are instances from the perspective of the away team.

```python
league_game_log = league_game_log[~league_game_log['MATCHUP'].str.contains('@')]
```

once you get all the game logs you can go and retrieve all playbyplay data. Usually when there are no rate limits I spam the API (WITHIN REASON) using multiprocessing however since the NBA api has some serious API limits this is a slow burning job that'll take a few hours. I suggest you run this before you get on with something else and let it run in the background.

```python
league_game_log
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEASON_ID</th>
      <th>TEAM_ID</th>
      <th>TEAM_ABBREVIATION</th>
      <th>TEAM_NAME</th>
      <th>GAME_ID</th>
      <th>GAME_DATE</th>
      <th>MATCHUP</th>
      <th>WL</th>
      <th>MIN</th>
      <th>FGM</th>
      <th>...</th>
      <th>DREB</th>
      <th>REB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
      <th>PLUS_MINUS</th>
      <th>VIDEO_AVAILABLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22013</td>
      <td>1610612747</td>
      <td>LAL</td>
      <td>Los Angeles Lakers</td>
      <td>0021300003</td>
      <td>2013-10-29</td>
      <td>LAL vs. LAC</td>
      <td>W</td>
      <td>240</td>
      <td>42</td>
      <td>...</td>
      <td>34</td>
      <td>52</td>
      <td>23</td>
      <td>8</td>
      <td>6</td>
      <td>19</td>
      <td>23</td>
      <td>116</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22013</td>
      <td>1610612748</td>
      <td>MIA</td>
      <td>Miami Heat</td>
      <td>0021300002</td>
      <td>2013-10-29</td>
      <td>MIA vs. CHI</td>
      <td>W</td>
      <td>240</td>
      <td>37</td>
      <td>...</td>
      <td>35</td>
      <td>40</td>
      <td>26</td>
      <td>10</td>
      <td>7</td>
      <td>20</td>
      <td>21</td>
      <td>107</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>22013</td>
      <td>1610612754</td>
      <td>IND</td>
      <td>Indiana Pacers</td>
      <td>0021300001</td>
      <td>2013-10-29</td>
      <td>IND vs. ORL</td>
      <td>W</td>
      <td>240</td>
      <td>34</td>
      <td>...</td>
      <td>34</td>
      <td>44</td>
      <td>17</td>
      <td>4</td>
      <td>18</td>
      <td>21</td>
      <td>13</td>
      <td>97</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>22013</td>
      <td>1610612755</td>
      <td>PHI</td>
      <td>Philadelphia 76ers</td>
      <td>0021300005</td>
      <td>2013-10-30</td>
      <td>PHI vs. MIA</td>
      <td>W</td>
      <td>240</td>
      <td>43</td>
      <td>...</td>
      <td>32</td>
      <td>40</td>
      <td>24</td>
      <td>16</td>
      <td>1</td>
      <td>18</td>
      <td>21</td>
      <td>114</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>22013</td>
      <td>1610612744</td>
      <td>GSW</td>
      <td>Golden State Warriors</td>
      <td>0021300017</td>
      <td>2013-10-30</td>
      <td>GSW vs. LAL</td>
      <td>W</td>
      <td>240</td>
      <td>46</td>
      <td>...</td>
      <td>41</td>
      <td>48</td>
      <td>34</td>
      <td>8</td>
      <td>9</td>
      <td>15</td>
      <td>22</td>
      <td>125</td>
      <td>31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1521</th>
      <td>22021</td>
      <td>1610612761</td>
      <td>TOR</td>
      <td>Toronto Raptors</td>
      <td>0022100784</td>
      <td>2022-02-01</td>
      <td>TOR vs. MIA</td>
      <td>W</td>
      <td>240</td>
      <td>39</td>
      <td>...</td>
      <td>28</td>
      <td>43</td>
      <td>20</td>
      <td>9</td>
      <td>3</td>
      <td>15</td>
      <td>24</td>
      <td>110</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1522</th>
      <td>22021</td>
      <td>1610612750</td>
      <td>MIN</td>
      <td>Minnesota Timberwolves</td>
      <td>0022100770</td>
      <td>2022-02-01</td>
      <td>MIN vs. DEN</td>
      <td>W</td>
      <td>240</td>
      <td>46</td>
      <td>...</td>
      <td>39</td>
      <td>52</td>
      <td>35</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>21</td>
      <td>130</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1525</th>
      <td>22021</td>
      <td>1610612741</td>
      <td>CHI</td>
      <td>Chicago Bulls</td>
      <td>0022100769</td>
      <td>2022-02-01</td>
      <td>CHI vs. ORL</td>
      <td>W</td>
      <td>240</td>
      <td>46</td>
      <td>...</td>
      <td>38</td>
      <td>49</td>
      <td>25</td>
      <td>5</td>
      <td>6</td>
      <td>10</td>
      <td>15</td>
      <td>126</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1526</th>
      <td>22021</td>
      <td>1610612759</td>
      <td>SAS</td>
      <td>San Antonio Spurs</td>
      <td>0022100771</td>
      <td>2022-02-01</td>
      <td>SAS vs. GSW</td>
      <td>L</td>
      <td>240</td>
      <td>46</td>
      <td>...</td>
      <td>29</td>
      <td>34</td>
      <td>33</td>
      <td>7</td>
      <td>7</td>
      <td>14</td>
      <td>17</td>
      <td>120</td>
      <td>-4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1528</th>
      <td>22021</td>
      <td>1610612756</td>
      <td>PHX</td>
      <td>Phoenix Suns</td>
      <td>0022100772</td>
      <td>2022-02-01</td>
      <td>PHX vs. BKN</td>
      <td>W</td>
      <td>240</td>
      <td>42</td>
      <td>...</td>
      <td>30</td>
      <td>37</td>
      <td>26</td>
      <td>8</td>
      <td>4</td>
      <td>16</td>
      <td>19</td>
      <td>121</td>
      <td>10</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10284 rows × 29 columns</p>
</div>



```python
from tqdm import tqdm
from time import sleep

pbp = []
for game_id in tqdm(league_game_log.GAME_ID):
    pbp.append(playbyplayv2.PlayByPlayV2(game_id).get_data_frames()[0])
    sleep(0.2)
    #to ensure over time we aren't spamming the api and hitting any rate limits, set it to whatever.
```

      0%|▏                                                                                                                                                                         | 14/10284 [00:09<1:52:42,  1.52it/s]


```python
df = pd.concat(pbp)
df.to_csv('bpbp.csv', index=False)
```

# Data Preprocessing and Featurizing 

okay we have our play by play data but now what? 
The first version of our model was to replicate what has already been done. So our current goal is to build 960 logistic regression models, one for each three second period.

Since our play by play data is not uniformly generated i.e. records have times from 0-2880 but they are not uniformly spaced and definitely not every three seconds (the shot clock itself goes on for 24 seconds) so what can we do? 

As Data Scientists real world data is never going to be perfect, formatting, quality, frequency etc but we must do what we can with what we have. 

And so we need to make assumptions and preprocess our data accordingly. 

Assumptions we're going to make:
- The state of the game is the same until the next play by play event i.e. if a play happened at 2700 and the score was 10-15 and the next play happened at 2680 and the score is now 12-15 then the time between 2700 and 2680 still has the score 10-15. 

The features we are aiming to produce are for the first iteration is

- Whos got possesion? 
- Score difference
- is it over time? 
