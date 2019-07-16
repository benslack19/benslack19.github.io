---
title: SILT - Substrings in SQL
categories: [data science]
---

Something I learned today was Substrings in SQL. 

Using my [favorite tutorial](https://sqlbolt.com/lesson/select_queries_introduction) I played around with the [substring function](https://www.techonthenet.com/sql_server/functions/substring.php). But depending on the server, the function might be known as "substr" which I discovered was necessary in the tutorial.

View the first 5 rows of the the table
`select * from movies limit 5;`

Get the first 5 characters from the director column
`select substr(director, 1, 5) from movies;`

Get 3 characters starting from position 5 characters from the director column
`select substr(director, 5, 3) from movies;`

Something harder would be to get the last 3 characters. I found [this post](https://stackoverflow.com/questions/8359772/t-sql-substring-last-3-characters) but needed to make some adaptations for the SQL style.
`SELECT SUBSTR(Director, length(Director)-2, 3) FROM movies`

If needed, the `SUBSTR(Director, length(Director)-2, 3)` portion could also be put in an ORDERBY statement.