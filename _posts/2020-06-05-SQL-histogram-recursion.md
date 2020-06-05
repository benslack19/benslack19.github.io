---
title: Histograms and recursion in SQL
toc: true
---

I came across a problem a few weeks ago about making a histogram in a SQL query. I did not expect to learn about recursion when I first started on this problem, but it's something I came across when working on this solution. For my example, I'll be using some baseball data, but this should work with whatever kind of data you have. The [SQL queries are done within a Jupyter notebook](https://benslack19.github.io/SQL-and-jupyter/).

[I don't care that much how it's done, just take me to the answer!](#final-query-fully-represented-histogram using recursion)


```python
import pandas as pd
import sqlalchemy
import sqlalchemy_utils
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
```


```python
# Define a database name
# Set your postgres username
dbname = "baseball"
username = "lacar"  # change this to your username

# Working with PostgreSQL in Python
# Connect to make queries using psycopg2
con = None
con = psycopg2.connect(database=dbname, user=username)

# Here, we're using postgres, but sqlalchemy can connect to other things too.
engine = create_engine("postgres://%s@localhost/%s" % (username, dbname))
print(engine.url)
```

    postgres://lacar@localhost/baseball


## Basic histogram

Let's say the problem is: **"Build a histogram for the number of at-bats for all major league baseball players in 2019. Place the counts into bins with width of 50-at-bats.** Therefore, count the number of players who had between 0 and 49 at-bats, 50 and 99 at-bats, 100 and 149, etc.

Let's get a preview of the table we're working with, ordering by the number of at-bats in descending fashion.


```python
# All MLB players
sql_query = """
SELECT "Name", "Team", "AB"
FROM batting_stats
WHERE "Season"=2019
ORDER BY "AB" DESC
LIMIT 5;
"""
pd.read_sql_query(sql_query,con)
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
      <th>Name</th>
      <th>Team</th>
      <th>AB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Whit Merrifield</td>
      <td>Royals</td>
      <td>681.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Marcus Semien</td>
      <td>Athletics</td>
      <td>657.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rafael Devers</td>
      <td>Red Sox</td>
      <td>647.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jonathan Villar</td>
      <td>Orioles</td>
      <td>642.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ozzie Albies</td>
      <td>Braves</td>
      <td>640.0</td>
    </tr>
  </tbody>
</table>
</div>



From this table, each player's number of at-bats is on a separate line. This makes the query pretty straightforward. Assuming we set it to bins of 50 at-bats, we can take each player's number of at-bats, divide by 50, and `FLOOR` the result. By then multiplying by 50, you can then get back the bin groups based on the original at-bats, and then use `COUNT` to produce the number in that bin.


```python
# All MLB players
sql_query = """
SELECT FLOOR("AB"/50.0)*50 AS ab_floor,
       COUNT(*)
FROM batting_stats
WHERE "Season"=2019
GROUP BY ab_floor
ORDER BY ab_floor;
"""
pd.read_sql_query(sql_query,con)
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
      <th>ab_floor</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50.0</td>
      <td>104</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100.0</td>
      <td>51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>150.0</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200.0</td>
      <td>51</td>
    </tr>
    <tr>
      <th>5</th>
      <td>250.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>6</th>
      <td>300.0</td>
      <td>46</td>
    </tr>
    <tr>
      <th>7</th>
      <td>350.0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>8</th>
      <td>400.0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>9</th>
      <td>450.0</td>
      <td>44</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500.0</td>
      <td>35</td>
    </tr>
    <tr>
      <th>11</th>
      <td>550.0</td>
      <td>37</td>
    </tr>
    <tr>
      <th>12</th>
      <td>600.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>13</th>
      <td>650.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



As a quick sanity check, we can look back up in the preview of our table and see that Whit Merrifield of the Royals and (681 at-bats) and Marcus Semien of the Athletics (657 at-bats) batted the most. This matches the count of "2" for the 650 at-bat bin. It makes sense that there are a lot of players in the 0-49 bin since pitchers don't hit in American League ballparks and many players have brief stints in the major leagues. Aside from these observations, I did more sanity checks when I worked this out on my own, so I think we can feel pretty good about the query.

But let's say we wanted to limit the histogram to [my favorite team](https://www.mlb.com/padres). We see something peculiar. 


```python
# Interim table - using Padres, doing a group by the bin
sql_query = """
SELECT FLOOR("AB"/50.0)*50 AS ab_floor,
       COUNT(*)
FROM batting_stats
WHERE "Season"=2019
AND "Team"='Padres'
GROUP BY ab_floor
ORDER BY ab_floor;
"""
df_query = pd.read_sql_query(sql_query,con)    
df_query
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
      <th>ab_floor</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>150.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>250.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>300.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>350.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>400.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>550.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>600.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



If you look closely, there are [missing bins!](https://media.giphy.com/media/iGvWZBfhOmBKEtWJmF/giphy.gif) The missing bins are where no players have at-bats in that bin. For example, no Padres player had an at-bat that fell between 450 and 500 at-bats, so that bin doesn't show up. But it would make a better histogram if all bins of 0 count are represented.

(We did not have to worry about this when we were looking at at-bats across 990 players from all 30 teams. With more players, "gaps" in the histogram are less likely. But since we limited this to the Padres only, this last query was with data from only 34 players.)

I searched around for tutorials and I couldn't seem to find an answer of how to include a bin and display a count of 0. [This tutorial](http://www.wagonhq.com/sql-tutorial/creating-a-histogram-sql) acknowledges the issue in the example they provided and had this quote:
> It has one failing in that if we have no data in a bucket (e.g. no purchases of 55 to 60 dollars), then that row will not appear in the results. We can fix that with a more complex query, but let’s skip it for now.

I did not see a more complex query later and the missing bin observation bugged me. This felt like a challenge :-)

## Recursion

I put on my clever hat and thought about what we can do. I thought about creating the bin intervals and then left joining the above query onto the manufactured bins. That seemed like a reasonable approach, but I hadn't used SQL to synthesize values before. That's where I came across [recursion in SQL](https://www.postgresqltutorial.com/postgresql-recursive-query/). Some of this syntax is specific to PostgreSQL so keep that in mind when using your SQL variant. 

From the tutorial, it looks like we would make a recursive CTE. What is interesting is how an SQL recursion differs from a Python recursion, at least from what I have learned. From a [Python recursion tutorial](https://runestone.academy/runestone/books/published/pythonds/Recursion/WhatIsRecursion.html), one starts with your input data and recursion shrinks this until a base case is met where the problem can be solved trivially. However in SQL recursion, you start with an "anchor term" (which appears analagous to a base case) and uses `UNION` or `UNION ALL` to horizontally concatenate the "recursive term" onto the anchor term. Recursion stops adding new rows when a condition is met.

It took me a while to wrap my head around this, so I started slow. My original goal is to create bins like 0, 50, 150, etc. but not rely on the data to give me back those values. First I looked at the range of data and how many bins I need.


```python
# Get number of bins
sql_query = """
SELECT MIN(FLOOR("AB"/50.0)*50) AS min_ab_bin,
       MAX(FLOOR("AB"/50.0)*50) AS max_ab_bin,
       (MAX(FLOOR("AB"/50.0)*50)-MIN(FLOOR("AB"/50.0)*50))/50 AS no_bins
FROM batting_stats
WHERE "Season"=2019
AND "Team"='Padres';
"""
df_query = pd.read_sql_query(sql_query,con)    
df_query
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
      <th>min_ab_bin</th>
      <th>max_ab_bin</th>
      <th>no_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>600.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>



I created something recursively but just using a placeholder as a bin number. The SQL recursion structure relies on creating a CTE using `WITH RECURSIVE`. My anchor term was simply 0. The recursive term was simply to add 1 and then it would stop when the bin_number was less than 13, since you can see from the above query that the max number of bins is 12.


```python
# Try doing something recursively
sql_query = """
WITH RECURSIVE
    no_bins_t AS
        
        (-- anchor term
        SELECT 0 AS bin_number
    
        UNION ALL
        
        --recursive term
        SELECT bin_number + 1 AS bin_number
        FROM no_bins_t
        WHERE bin_number < 13)

SELECT bin_number
FROM no_bins_t;
"""
df_query = pd.read_sql_query(sql_query,con)    
df_query
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
      <th>bin_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>



Great. Now we can put in proper subqueries and variables instead of relying on hard coding numbers. The anchor term has the subquery that evaluates the minimum bin. The 13 in the recursive term can be substituted for with the subquery by identifying the max bin.


```python
# Get range
sql_query = """
WITH RECURSIVE
    no_bins_t AS
        
        (-- anchor term
        SELECT MIN(FLOOR("AB"/50.0)*50) AS bin_number
        FROM batting_stats
        WHERE "Season"=2019
        AND "Team"='Padres'
    
        UNION ALL
        
        --recursive term
        SELECT bin_number + 1 AS bin_number
        FROM no_bins_t
        WHERE bin_number < (SELECT MAX(FLOOR("AB"/50.0)) + 1
                            FROM batting_stats
                            WHERE "Season"=2019
                            AND "Team"='Padres'))

SELECT bin_number,
       bin_number*50 AS at_bat_bin
FROM no_bins_t;
"""
df_query = pd.read_sql_query(sql_query,con)    
df_query
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
      <th>bin_number</th>
      <th>at_bat_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>250.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.0</td>
      <td>300.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.0</td>
      <td>400.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>450.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.0</td>
      <td>500.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11.0</td>
      <td>550.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12.0</td>
      <td>600.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13.0</td>
      <td>650.0</td>
    </tr>
  </tbody>
</table>
</div>



## Final query: fully represented histogram using recursion


```python
# Get range
sql_query = """
WITH RECURSIVE
    no_bins_t AS
        
        (-- anchor term
        SELECT MIN(FLOOR("AB"/50.0)*50) AS bin_number
        FROM batting_stats
        WHERE "Season"=2019
        AND "Team"='Padres'
    
        UNION ALL
        
        --recursive term
        SELECT bin_number + 1 AS bin_number
        FROM no_bins_t
        WHERE bin_number < (SELECT MAX(FLOOR("AB"/50.0)) + 1
                            FROM batting_stats
                            WHERE "Season"=2019
                            AND "Team"='Padres'))

SELECT bin_number,
       bin_number*50 AS at_bat_bin,
       n_players
FROM no_bins_t

LEFT JOIN

(SELECT FLOOR("AB"/50.0)*50 AS ab_floor,
        COUNT(*) AS n_players
FROM batting_stats
WHERE "Season"=2019
AND "Team"='Padres'
GROUP BY ab_floor
ORDER BY ab_floor) AS padres_ab

ON no_bins_t.bin_number*50=padres_ab.ab_floor;
"""
df_query = pd.read_sql_query(sql_query,con)    
df_query
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
      <th>bin_number</th>
      <th>at_bat_bin</th>
      <th>n_players</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>50.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>100.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>150.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>200.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>250.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.0</td>
      <td>300.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>350.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.0</td>
      <td>400.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>450.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.0</td>
      <td>500.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11.0</td>
      <td>550.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12.0</td>
      <td>600.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13.0</td>
      <td>650.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We have `NULL` values but we can easily clean this up using CASE.


```python
# Get range
sql_query = """
WITH RECURSIVE
    no_bins_t AS
        
        (-- anchor term
        SELECT MIN(FLOOR("AB"/50.0)*50) AS bin_number
        FROM batting_stats
        WHERE "Season"=2019
        AND "Team"='Padres'
    
        UNION ALL
        
        --recursive term
        SELECT bin_number + 1 AS bin_number
        FROM no_bins_t
        WHERE bin_number < (SELECT MAX(FLOOR("AB"/50.0)) + 1
                            FROM batting_stats
                            WHERE "Season"=2019
                            AND "Team"='Padres'))

SELECT bin_number,
       bin_number*50 AS at_bat_bin,
       CASE WHEN n_players IS NULL THEN 0
            ELSE n_players END AS n_players
FROM no_bins_t

LEFT JOIN

(SELECT FLOOR("AB"/50.0)*50 AS ab_floor,
        COUNT(*) AS n_players
FROM batting_stats
WHERE "Season"=2019
AND "Team"='Padres'
GROUP BY ab_floor
ORDER BY ab_floor) AS padres_ab

ON no_bins_t.bin_number*50=padres_ab.ab_floor;
"""
df_query = pd.read_sql_query(sql_query,con)    
df_query
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
      <th>bin_number</th>
      <th>at_bat_bin</th>
      <th>n_players</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>50.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>100.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>150.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>200.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>250.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.0</td>
      <td>300.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>350.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.0</td>
      <td>400.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>450.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.0</td>
      <td>500.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11.0</td>
      <td>550.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12.0</td>
      <td>600.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13.0</td>
      <td>650.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We can make things look nicer with `CONCAT`, although we need to rely on ordering by bin_number since the concatenated at_bat_bin will look like a string and some will appear out of order.


```python
# Get range
sql_query = """
WITH RECURSIVE
    no_bins_t AS
        
        (-- anchor term
        SELECT MIN(FLOOR("AB"/50.0)*50) AS bin_number
        FROM batting_stats
        WHERE "Season"=2019
        AND "Team"='Padres'
    
        UNION ALL
        
        --recursive term
        SELECT bin_number + 1 AS bin_number
        FROM no_bins_t
        WHERE bin_number < (SELECT MAX(FLOOR("AB"/50.0)) + 1
                            FROM batting_stats
                            WHERE "Season"=2019
                            AND "Team"='Padres'))

SELECT bin_number,
       CONCAT(bin_number*50, '-', bin_number*50 + 49) AS at_bat_bin_interval,
       CASE WHEN n_players IS NULL THEN 0
            ELSE n_players END AS n_players
FROM no_bins_t

LEFT JOIN

(SELECT FLOOR("AB"/50.0)*50 AS ab_floor,
        COUNT(*) AS n_players
FROM batting_stats
WHERE "Season"=2019
AND "Team"='Padres'
GROUP BY ab_floor
ORDER BY ab_floor) AS padres_ab

ON no_bins_t.bin_number*50=padres_ab.ab_floor
ORDER BY no_bins_t.bin_number;

"""
df_query = pd.read_sql_query(sql_query,con)    
df_query
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
      <th>bin_number</th>
      <th>at_bat_bin_interval</th>
      <th>n_players</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0-49</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>50-99</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>100-149</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>150-199</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>200-249</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>250-299</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.0</td>
      <td>300-349</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>350-399</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.0</td>
      <td>400-449</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>450-499</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.0</td>
      <td>500-549</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11.0</td>
      <td>550-599</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12.0</td>
      <td>600-649</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13.0</td>
      <td>650-699</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



There you have it. One thing to keep in mind is that the recursive query uses a CTE and I haven't yet figured out a way to use another CTE at the same time. If you figure this out, please let me know!