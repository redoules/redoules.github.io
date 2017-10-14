# Notes on Data Science, Machine Learning, and Artificial Intelligence

I am a [data scientist originally trained as a quantitative political scientist](http://chrisalbon.com/pages/about.html). My career has focused on applying data science to political and social issues. Years ago, I became frustrated at a major gap in the existing literature. On one side sat data science, with roots in mathematics and computer science. On the other side was the social sciences, with hard-earned expertise modeling and predicting complex human behavior. However, despite each field having much to lend to the other, collaborations between the two have been exceptionally rare.

The motivation of this ongoing book project is to bridge that gap: to create a practical guide to applying data science to political and social phenomena.

## Writing Content Procedure
- Write in Jupyter or markdown
- Add to correct folder
- Add YAML matter:
    Title: Beautiful Soup Basic HTML Scraping  
    Slug: beautiful_soup_html_basics  
    Summary: Beautiful Soup Basic HTML Scraping  
    Date: 2016-05-01 12:00  
    Category: Python  
    Tags: Web Scraping    
    Authors: Chris Albon  
- Change `![png](...)` to `![png]({filename}/images/scipy_simple_clustering/output_26_1.png)`
- Move post images in their folder to content/images/{filename}
- Run Pelican
- View Page

## Deployment Procedure

- pelican content -o output -s pelicanconf.py
- ghp-import output
- git push origin gh-pages

## Development Procedure

- cd output
- python -m pelican.server
- pelican

## To Do List

- Rewrite bio / CV
- add comments to css
