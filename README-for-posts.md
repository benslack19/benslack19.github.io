# Process for generating a data science blog post

- Make a new branch: `git checkout -b new-post`
- Create post in a notebook (`.ipynb` file) and save to the `_posts_drafts` directory. Prefix the file name with the date in the form `YYYY-MM-DD`.
- Add yaml matter in the first cell. Here's an example:
```
---
title: "Generating a predictive distribution for the number of people attending your party"
mathjax: true
toc: true
toc_sticky: true
categories: [data science, statistics]
---
```
- Run the `nb2md_script.sh` script with the `.ipynb` file as an argument. This will create a markdown file in `_posts` and any figures saved to `assets`.
- Open the markdown file and add back three dashes `---` to the first line.
- Run `bundle exec jekyll serve` and verify the page looks good locally.
- Merge the branch into main and push: `git checkout main && git merge new-post && git push`
