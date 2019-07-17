---
title: SILT: Opening files and file handles in Python
categories: [data science]
---

Something I learned today was files and file handles in Python. 

Actually it's more of "something I keep forgetting about and have to look up on Stack Overflow because I don't use it enough".

I use pandas a ton and don't always read in data using file handles so thought it'd be good to review and document. The way I've liked to open files from a file handle is using `with` since I don't have to deal with closing the file. [Here](https://stackoverflow.com/questions/40096612/how-do-i-open-a-text-file-in-python) is one post that uses with.


`with open 

with open('my_text_file.txt', 'r') as fhand:
    for line in fhand:
        print(line)
`

`fhand` represents the file handle. I like how [Dr. Chuck](http://www.dr-chuck.com) explains file handles [here](https://www.py4e.com/html3/07-files).

*The file handle is not the actual data contained in the file, but instead it is a "handle" that we can use to read the data. You are given a handle if the requested file exists and you have the proper permissions to read the file.*