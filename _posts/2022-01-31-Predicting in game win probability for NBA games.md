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

# Data - Gathering/Processing/Cleaning

As mentioned before, python has a GREAT wrapper for the stats.nba.com api again linked [here](https://github.com/swar/nba_api), which is worth checking out in your own time just to see the volume of data available to play with. But I wrote a simple script to collect all the playbyplay data for the last ~7 odd years. 
The problem is rate limits! 

```python
#script to get all live playbyplay data
from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import tqdm
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

The reason i'm filtering for the home games below is because there can only be two outcomes in basketball, win or lose and so if you find the probability that the home team wins then you find the probability that the away team wins too so we focus our modelling efforts on the home team winning

```python
league_game_log = league_game_log[~league_game_log['MATCHUP'].str.contains('@')]
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
      <th>FGA</th>
      <th>FG_PCT</th>
      <th>FG3M</th>
      <th>FG3A</th>
      <th>FG3_PCT</th>
      <th>FTM</th>
      <th>FTA</th>
      <th>FT_PCT</th>
      <th>OREB</th>
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
      <td>93</td>
      <td>0.452</td>
      <td>14</td>
      <td>29</td>
      <td>0.483</td>
      <td>18</td>
      <td>28</td>
      <td>0.643</td>
      <td>18</td>
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
      <th>1</th>
      <td>22013</td>
      <td>1610612746</td>
      <td>LAC</td>
      <td>Los Angeles Clippers</td>
      <td>0021300003</td>
      <td>2013-10-29</td>
      <td>LAC @ LAL</td>
      <td>L</td>
      <td>240</td>
      <td>41</td>
      <td>83</td>
      <td>0.494</td>
      <td>8</td>
      <td>21</td>
      <td>0.381</td>
      <td>13</td>
      <td>23</td>
      <td>0.565</td>
      <td>10</td>
      <td>30</td>
      <td>40</td>
      <td>27</td>
      <td>11</td>
      <td>4</td>
      <td>16</td>
      <td>21</td>
      <td>103</td>
      <td>-13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22013</td>
      <td>1610612741</td>
      <td>CHI</td>
      <td>Chicago Bulls</td>
      <td>0021300002</td>
      <td>2013-10-29</td>
      <td>CHI @ MIA</td>
      <td>L</td>
      <td>240</td>
      <td>35</td>
      <td>83</td>
      <td>0.422</td>
      <td>7</td>
      <td>26</td>
      <td>0.269</td>
      <td>18</td>
      <td>23</td>
      <td>0.783</td>
      <td>11</td>
      <td>30</td>
      <td>41</td>
      <td>23</td>
      <td>11</td>
      <td>4</td>
      <td>19</td>
      <td>27</td>
      <td>95</td>
      <td>-12</td>
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
      <td>72</td>
      <td>0.514</td>
      <td>11</td>
      <td>20</td>
      <td>0.550</td>
      <td>22</td>
      <td>29</td>
      <td>0.759</td>
      <td>5</td>
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
      <th>4</th>
      <td>22013</td>
      <td>1610612753</td>
      <td>ORL</td>
      <td>Orlando Magic</td>
      <td>0021300001</td>
      <td>2013-10-29</td>
      <td>ORL @ IND</td>
      <td>L</td>
      <td>240</td>
      <td>36</td>
      <td>93</td>
      <td>0.387</td>
      <td>9</td>
      <td>19</td>
      <td>0.474</td>
      <td>6</td>
      <td>10</td>
      <td>0.600</td>
      <td>13</td>
      <td>26</td>
      <td>39</td>
      <td>17</td>
      <td>10</td>
      <td>6</td>
      <td>19</td>
      <td>26</td>
      <td>87</td>
      <td>-10</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Why did I do the abov
