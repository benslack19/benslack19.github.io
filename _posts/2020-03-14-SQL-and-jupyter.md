---
title: PostgreSQL and Jupyter notebooks
toc: true
---

PostgreSQL is one of the more popular flavors of SQL. A common way to use PostgreSQL is with pgadmin but I do not think their [UI](https://www.pgadmin.org/screenshots/) is the most user-friendly. By contrast, interacting with PostgreSQL through a Jupyter notebook has several benefits. The most immediately apparent benefit is that SQL queries can be brought directly into pandas for data analysis, visualization, and machine learning. But the process of learning SQL itself is supported by the notebook structure. Notebooks are inherently good formats for note-taking, such as with markdown. In addition, with different cells in the same notebook, one can build complicated queries by first writing, understanding, and having a record of simple queries.

Here is an example of a complicated looking query, based on baseball data I collected for my Insight Data Science Fellowship project. (For the baseball aficionados, the question I asked was "What is each team's hit average against the infield shift versus a standard alignment?" The query is not perfect for a performance metric but this gives me a decent approximation.)

```
sql_query = """
SELECT 
    ob_events_table.team_name,
    ob_events_table.if_fielding_alignment,
    ob_events_table.n_ob,
    total_ab_table.n_ob,
    ob_events_table.n_ob::decimal/total_ab_table.n_ob AS obp
FROM
    (SELECT 
        CASE WHEN inning_topbot='Top' THEN away_team
             WHEN inning_topbot='Bot' THEN home_team
             END AS team_name,
        if_fielding_alignment,
        COUNT(batter) AS n_ob
    FROM statcast
    WHERE events IS NOT NULL
    AND events IN ('single', 'double', 'triple', 'home_run')
    GROUP BY team_name, if_fielding_alignment
    HAVING if_fielding_alignment IN ('Infield shift', 'Standard')) AS ob_events_table
JOIN
    (SELECT 
        CASE WHEN inning_topbot='Top' THEN away_team
             WHEN inning_topbot='Bot' THEN home_team
             END AS team_name,
        if_fielding_alignment,
        COUNT(batter) AS n_ob
    FROM statcast
    WHERE events IS NOT NULL
    GROUP BY team_name, if_fielding_alignment
    HAVING if_fielding_alignment IN ('Infield shift', 'Standard')) AS total_ab_table
ON
    ob_events_table.team_name=total_ab_table.team_name
    AND
    ob_events_table.if_fielding_alignment=total_ab_table.if_fielding_alignment;
"""
df_query = pd.read_sql_query(sql_query,con)
print(df_query)
```

```
# code block test
test = x + 1
```

~~~~{.python}
# code block test with tildes
test = x + 1
~~~~


You can find the answer, see how I built this query, and find ways to setup PostgreSQL with Jupyter yourself on my Insight project repo located [here](https://github.com/benslack19/baseball_player_selector/blob/master/SQLqueries_postgreSQL.ipynb).