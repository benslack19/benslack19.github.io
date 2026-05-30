# benslack19.github.io

Personal website including blog, portfolio, and notes, based on the Jekyll theme [minimal-mistakes](https://mmistakes.github.io/minimal-mistakes/).

## Local development

```bash
bundle install
bundle exec jekyll serve
```

The site will be available at `http://localhost:4000`.

## Creating a blog post

1. Create a new branch: `git checkout -b new-post`
2. Write the post as a Jupyter notebook (`.ipynb`) in `_posts_drafts/`. Prefix the filename with the date: `YYYY-MM-DD-post-title.ipynb`.
3. Add YAML front matter in the first notebook cell:

   ```yaml
   ---
   title: "Your Post Title"
   mathjax: true
   toc: true
   toc_sticky: true
   categories: [data science, statistics]
   ---
   ```

4. Run the conversion script to generate markdown and copy figures to `assets/`:

   ```bash
   ./_posts_drafts/nb2md_script.sh _posts_drafts/YYYY-MM-DD-post-title.ipynb
   ```

5. Open the generated markdown file in `_posts/` and add `---` to the first line.
6. Preview locally with `bundle exec jekyll serve` and verify the post looks correct.
7. Merge to main and push:

   ```bash
   git checkout main && git merge new-post && git push
   ```
