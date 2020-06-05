---
title: PostgreSQL and Jupyter notebooks
toc: false
---

[PostgreSQL](https://www.postgresql.org) is one of the most popular variants of SQL. It is common to use PostgreSQL with pgadmin but I am not a big fan of their [UI](https://www.pgadmin.org/screenshots/). By contrast, interacting with PostgreSQL through a [Jupyter notebook](https://jupyter-notebook.readthedocs.io/en/stable/) has several benefits. Most immediately apparent is that SQL queries can be brought directly into a Python environment for data analysis, visualization, and machine learning. But the process of learning SQL itself is supported by the notebook format. Jupyter make it easy to adopt [literate programming](https://en.wikipedia.org/wiki/Literate_programming) practices since note-taking is easy. In addition, with different cells in the same notebook, one can build complex SQL queries by first writing, understanding, and recording simpler commands.

Here is an example of a complicated-looking query in a Jupyter notebook cell. This is based on data for [my baseball project](https://github.com/benslack19/baseball_player_selector) that I worked on as a fellow at [Insight Data Science](https://www.insightdatascience.com). (For the baseball aficionados, the question I asked was "What is each team's hit average against the infield shift versus a standard alignment?" The output provides an approximation of team batting average.)

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
```

You can see the output of this query (shift against the Angels and Orioles!), see how I built it, and learn how to setup PostgreSQL with Jupyter yourself in [this notebook](https://github.com/benslack19/baseball_player_selector/blob/master/SQLqueries_postgreSQL.ipynb) located on my project repo.