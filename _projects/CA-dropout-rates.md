---
title: Trends in CA school dropouts
excerpt: I looked at public school and demographic data on high school student dropouts to apply and practice Python data munging and visualization skills.
header:
  image: /assets/edstudy/school-417612_1920.jpg
  teaser: /assets/edstudy/school-417612_1920.jpg
gallery:
  - url: /assets/edstudy/output_52_1.png
    image_path: /assets/edstudy/output_52_1.png
    alt: "Dropout rate percentage by ethnicity"
  - url: /assets/edstudy/output_60_1.png
    image_path: /assets/edstudy/output_60_1.png
    alt: "Dropout rate percentage by school median income"
  - url: /assets/edstudy/output_66_1.png
    image_path: /assets/edstudy/output_66_1.png
    alt: "Dropout rate percentage by school type"
---

Receiving credit for Coursera's course [Applied Plotting, Charting & Data Representation in Python](https://www.coursera.org/learn/python-plotting) requires completion of a small data visualization project. I wanted further practice in application so I chose to look at data from a subject that I'm passionate about. I analyzed year 2015-16 data on high school student dropouts which is available from the California Department of Education [website](https://www.cde.ca.gov/ds/sd/sd/filesdropouts.asp) and obtained supplementary information from the [United States Census Bureau](https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml).

They were good datasets as practice for what I learned in the course. Doing the project required use of a variety of data importing, munging, and visualization techniques. For web scraping and data importing, I used urllib, io, and beautifulsoup. For data munging in pandas, I performed merging and filtering and used group-by and pivot table methods. For data visualization, I used Matplotlib and Seaborn to create scatter plots, bar plots, and histograms. This is done while espousing the figure visualization philosophies of [Alberto Cairo](http://www.thefunctionalart.com) and [Edward Tufte](https://www.edwardtufte.com/tufte/) which were promoted in the Coursera course. While some students of the course thought this was a wasted lecture, I learned to appreciate good figure designs. In addition, my project on dropout rates incorporated some statistical analysis using SciPy and unsupervised machine learning (PCA) using scikit-learn.

However, what I have revealed in my project thus far is not novel nor unexpected. In fact, not long after I embarked on this personal project, I saw that kidsdata.org had performed [similar analysis](http://www.kidsdata.org/topic/106/highschooldropouts-race/table#fmt=193&loc=2&tf=84&ch=7&sortColumnId=0&sortType=asc) with the data. This was not necessarily a bad thing since I was able to use their work as a sanity check for some of the analysis that I did.

To start, I had to make some assumptions and decisions on the scope of the study. First, I chose to limit my analysis to high schools that had at least 25 students each in grades 9-12. Therefore, high schools that did not have a 9th grade or schools that had a very small number of students were filtered out. In total, there remained 1,804,448 students from 1,352 schools to which to do the analysis. It was also important to me to include economic characteristics of each school. However, I could not find school-specific information on family income. Instead, I used the census data which provides median income by zip code as a proxy. Therefore, two high schools within the same zip code could have students from different median income levels, but would be represented by the same median income in the study. It is my assumption that this situation does not occur often but it is a caveat worth noting. (While investigating the diversity of income levels, it was eye-opening to see the inequity in [wealth distribution](/assets/edstudy/output_37_1.png).)

{% include gallery caption="Gallery of data visualization figures for this project." %}

The figures in the gallery are a sampling of the project. The first figure shows dropout rate percentage by ethnicity. Statewide, the dropout percentage for all students is ~1.5%. However, minorities such as African-Americans, Hispanics, and Native American and Alaska Natives are dropping out at a rate higher than this average. Interestingly, the students who chose not to report their ethnicity showed a far higher rate of dropping out than any other category. However, when looking at total enrollment](/assets/edstudy/output_37_1.png]), the "Not reported" category of students is very small relative to the others. The second figure highlights dropout rate percentage by local school median income. Schools located in the poorest areas are the most at-risk for having a high dropout percentage. The dataset also allowed for looking at dropouts by school type, which is something that I had not initially thought to investigate. The vast majority (94.2%) of CA high school students are enrolled in public high schools, of which a very high proportion show very low dropout rates. On the other hand, juvenile court schools and county community schools show very high dropout rates. (I just happened to catch up with a high school friend recently who teaches at a juvenile court school and he gave me some insight into some of the challenges there.)

A basic mission of our school system is to see that all students graduate high school and avoid dropping out. Students who drop out face significant life challenges, including lower wages and higher incarceration rates. However, the factors that influence a school's dropout rate are numerous and complex. The purpose of this data science project is to explore and identify features that could contribute to high dropout rate. What I've shown here is only a small portion of what correlates with student dropouts.

Importantly, there are many school factors that counteract apparent trends in dropout rates. What is particularly encouraging is the high number of schools that seem "at-risk" yet have low dropout rates. For example, in the figure showing dropout rate by local median income, half of the schools that are in the poorest areas CA still have dropout rates in the low single digits. What is special about schools with low dropout rates despite socio-economic challenges? Some schools are even performing exceptionally. Stockton Collegiate International Secondary School resides in a zip code with a local median income of $14,579 but yet has only 0.42% of students dropping out. Not surprisingly, they have earned accolades in US News & World Report according to their [website](http://www.stocktoncollegiate.org). Studying schools like this and performing additional analysis could help reveal factors that help schools in challenging circumstances.

While the main purpose of this project is to apply data science tools, there's clearly more to address much remains to be learned in this complicated issue of student dropouts.

For this project, my code and additional figures can be found in a Jupyter notebook on my Github page [here](https://github.com/benslack19/CA_dropout_rates/blob/master/EdDataAnalysis_OverperformingSchools.ipynb).
