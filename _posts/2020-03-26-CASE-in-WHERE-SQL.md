---
title: Using CASE in the WHERE statement of SQL
toc: true
categories: [data science, SQL]
---

## Problem statement

I was working on a baseball query when I discovered that I could not use a simple WHERE statement to select Padres at-bats. However, I knew the information was there. The teams were under the "away_team" and "home_team" columns. Here's what I was working with:

```
batter	game_date	events	inning_topbot	away_team	home_team
0	544725.0	2019-07-19	double	Top	CWS	TB
1	641553.0	2019-07-19	single	Top	CWS	TB
2	570560.0	2019-07-19	field_out	Top	CWS	TB
3	445055.0	2019-07-19	single	Top	CWS	TB
4	602922.0	2019-07-19	strikeout	Top	CWS	TB
```

I could use the "inning_topbot" as a conditional for my selection. I had used CASE in a SELECT statement before, but luckily I realized CASE can be used in a variety of statements including with WHERE. Here is the query that gave me what I wanted. (I'm doing this in a Jupyter notebook hence some of the pandas syntax.)

## Working query

```
sql_query = """
SELECT    
    batter, 
    game_date, 
    events,
    inning_topbot,
    away_team,
    home_team
FROM statcast

--I'm doing multiple filtering statements
WHERE "events" IS NOT NULL
AND "game_date" > '2019-03-03'
AND 

--Here is the CASE line
    CASE WHEN "inning_topbot"='Top' THEN away_team='SD'
         ELSE home_team='SD' END

--I ordered by random so I can see different examples and confirm my query
ORDER BY RANDOM();
"""
df_query = pd.read_sql_query(sql_query,con)
df_query
```

Here are the heads and tails of the output:

```
batter	game_date	events	inning_topbot	away_team	home_team
0	614177.0	2019-05-02	single	Top	SD	ATL
1	595978.0	2019-07-26	strikeout	Bot	SF	SD
2	571976.0	2019-03-31	strikeout	Bot	SF	SD
3	595978.0	2019-05-31	strikeout	Bot	MIA	SD
4	664119.0	2019-08-26	double	Bot	LAD	SD
...	...	...	...	...	...	...
6006	641778.0	2019-04-14	strikeout	Top	SD	ARI
6007	595978.0	2019-05-25	strikeout	Top	SD	TOR
6008	665487.0	2019-04-11	sac_fly	Top	SD	ARI
6009	614177.0	2019-05-04	strikeout	Bot	LAD	SD
6010	642336.0	2019-08-19	field_out	Top	SD	CIN
```