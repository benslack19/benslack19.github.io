---
title: School and community features that promote low income student success
excerpt:  I investigated the variables of high schools and their local areas which are associated with low income student college eligiblity.
toc: true
header:
  image: /assets/edstudy/school-417612_1920.jpg
  teaser: /assets/edstudy/school-417612_1920.jpg
gallery:
  - url: /assets/edstudy/output_52_1.png
    image_path: /assets/edstudy/output_52_1.png
    alt: "Dropout percentage by ethnicity"
  - url: /assets/edstudy/output_60_1.png
    image_path: /assets/edstudy/output_60_1.png
    alt: "Dropout percentage by school median income"
  - url: /assets/edstudy/output_66_1.png
    image_path: /assets/edstudy/output_66_1.png
    alt: "Dropout percentage by school type"
---

Low income students face challenges academically which affects long-term employment and upward social mobility. Fortunately some schools and communities have provided environments for low income students to thrive. In this project, I investigated the features of California (CA) public high schools and their local areas which are associated with the school's percentage of low income students that are college eligible (SPLICE). To obtain school-specific features, I obtained data from GreatSchools websites and from the CA public schools database. This was supplemented with local economic characteristics from publicly available census data. Identifying important features was obtained with random forest regression models. I find that a school's college eligibility rate for all students is a significantly good predictor for low income student performance. This highlights how the right environment benefits all demographics of students. Additionally, charter status is also an important variable, with charter schools positively correlating with SPLICE. Additional statistical analysis was performed following segmentation by the highest and lowest performing schools for SPLICE. Communities with a greater proportion of workers in professional, information, or manufacturing industries were positively associated with SPLICE while the percentage of a community that was not in the labor force was negatively correlated with SPLICE. Features that had no impact on SPLICE included the percentage of students that are low income students, magnet status, student-to-teacher ratio, and a community's median household income. This study may add to further investigation of actionable factors that can facilitate low income student success.

## Introduction

My interest in education stems from my view that education can be a driver of socioeconomic mobility. While there is mixed success on how much education may actually affect this (reference), I wanted to investigate how well low income students were doing and what school and community factors might drive how well they are doing.

I also focused on CA for several reasons. First, I am familiar with the state including the biggest metropolitan areas. I grew up in San Diego and lived in multiple areas. I went to college in West Los Angeles and tutored in South Los Angeles. I currently live in the Bay Area. The geographic domain knowledge helped me check some assumptions. Second, California is home to the largest state population in the US and has multiple, diverse socioeconomic locales. The coast tends to be more affluent and contains the biggest cities. Further inland tends to be rural with more farmland. There’s a variety of demographics with a range of ethnic diversity. Political affiliations tend to be correlated. This diversity is also reflected in the school and district composition: arge districts (LA), small schools, various enrollment sizes, etc.

While I am interested in understanding academic success at the individual student level, obtaining data for individual students was not practical. However, I could get access to the CA public schools database, GreatSchools information, and the census information. The school databases were on the level of schools, but the census information had the most granularity at the zip code level. Therefore, I reframed my success criteria as what proportion of students at a school are doing well? And what are the factors associated with that community and school that could drive that success? What does “doing well mean”? From the data, I had looked at dropout rates and high school graduation rates, but most schools were either very close to a proportion of 0 or 1, respectively. This posed some data science and statistical challenges (some common statistical assumptions assume Gaussian distributions or evaluation of proportions that are not close to the 0,1 boundaries). But also practically, there is not enough variability in those targets to really uncover something meaningful. However, a different metric that I thought could be meaningful is what proportion of low income students that are eligible for UC/CSU universities.

To be eligible for University of California or California State System colleges, a high school student must achieve a grade of C or better in two years of history/social science, English (4 years), math (3 years), a laboratory science (2 years), a language other than English (2 years), visual and performing arts (1 year), and a college preparatory elective. I felt this was a reasonable bar to clear as a definition of success.Additionally, I could also look at school and community features that explain a gap in success between low and non-low income students. I did this with a separate model…. This can be critical in helping to explain where and why some students might fail.

I obtained as many features as possible but I had to filter out many of them as described below. This was a project that was as much an exercise in data science as it was in uncovering what we can learn about education for low income students in California.

## Methods

### Data collection
Data was collected through three sources: (1) CA economic characteristics on the zip code level, (2) the CA Public Schools Database, and (3) GreatSchools. California economic characteristics was taken from

Data was collected through three sources: (1) CA economic characteristics on the zip code level, (2) the CA Public Schools Database, and (3) GreatSchools. California economic characteristics was taken from the 2015-16 census data (ref). This includes features of a community at the zip code level including the types of industries, what income benefits people have, health insurance status, employment status, worker class, and work commute. The majority of features included for analysis are on a percentage basis so that communities can be compared. For data preparation, text was edited to make column headers more human-readable. Zip codes where more than 90% of census data was missing was removed.

The CA Public Schools database provided school information such as name, district, identification numbers, county, address, and enrollment size. School name, district, and zip code were used for merging datasets. Enrollment size was used to filter data and derive other metrics. Only active, public high schools from this database were included for further data collection and analysis.

Information from GreatSchools was used to obtain school-specific performance and data about school staff. An API provided by GreatSchools was used to obtain some high-level metrics via an XML tree. Additional metrics, including student-to-counselor ratio, student-to-teacher ratio, average teacher salary, test scores, graduation rates, and our target variable (SPLICE), was through web scraping publicly available data.

After data collection, the dataset contained 1110 schools and 264 features. Additional feature engineering and selection from all three sources is described below and in detail in the notebook here.

###  Data preprocessing and feature selection

I chose to use tree-based models to predict SPLICE to guard against restrictions required by linear models, such as additivity (that variables are independent of each other) and that the relationship between variables and the predictors are linear. Nevertheless, I further filtered the schools and features to improve model robustness and interpretability. Since SPLICE is a metric of college eligibility, I only included schools that had a reasonable number of enrolled seniors and low income seniors. Restricting the schools with at least 25 total twelfth-graders with at least 10 being left 966 schools.

A number of steps were taken for feature (variable) selection. First, a feature was eliminated if missing values were present in at least ⅓ of the schools. Second, steps were taken to reduce multi-collinearity as indicated by variance inflation factor. This was present for many variables if they were a percentage of a category. Therefore some members of the category were dropped. For example, worker class contained a percentage of workers who were private wage, government, self-employed or unpaid. In this case, the percentage of workers in government were dropped since they were highly anti-correlated with the percentage of workers with private wages. I also dropped unpaid family workers since they were rare. Additional correlations were identified by taking the Pearson correlation and/or with scatter plots.

I also employed feature engineering and preparation. The number of enrolled seniors, SPLICE and the percentage of all students that are college eligible was used to estimate the percentage of non-low-income students that are college eligible. Ethnic composition of a school was also subject to multicollinearity, with the percentage of Cauasians and Hispanics negatively correlated. Ethnic representation in the model is indicated by the percentage of underrepresented minorities (African American, American Indian or Hispanic, as indicated here). A list of all original features that were removed can be found with the retained features that they correlated with (here)[link to a table].

I performed one last filtering step, keeping schools where no missing values were present in all remaining 70 features, leaving me with 814 school.

### Data visualization

To explore my data, I applied visualization packages matplotlib and seaborn for figures and geopandas for geographic data. In some cases, industry was aggregated to the county level.


# Results

Since I'm interested in education as a driver of societal change, I wondered what schools are doing well in the face of apparently adverse circumstances. Therefore, I explored trends of high school student dropout data from 2015-16 which is available from the California Department of Education [website](https://www.cde.ca.gov/ds/sd/sd/filesdropouts.asp). I also obtained supplementary information from the [United States Census Bureau](https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml).

Most of what I have revealed in this is not novel nor unexpected. In fact, not long after I embarked on this personal project, I saw that kidsdata.org had performed [similar analysis](http://www.kidsdata.org/topic/106/highschooldropouts-race/table#fmt=193&loc=2&tf=84&ch=7&sortColumnId=0&sortType=asc) with the data. This was not necessarily a bad thing since I was able to use their work as a sanity check for some of the analysis that I did.

Nevertheless, exploring the data myself allowed me to hone in on my question. To start, I made some assumptions and decisions on the scope of the study. First, I chose to limit my analysis to high schools that had at least 25 students each in grades 9-12. Therefore, high schools that did not have a 9th grade or schools that had a very small number of students were filtered out. In total, there remained 1,804,448 students from 1,352 schools on which to do the analysis. It was also important to me to include economic characteristics of each school. However, I could not find school-specific information on income. Instead, I used the census data which provides median income by zip code as a proxy. Therefore, two high schools within the same zip code could have students from different median income levels, but they would be represented by the same median income in the study. It is my assumption that this situation does not occur often but it is a caveat worth noting. (While investigating the diversity of income levels, it was eye-opening to see the inequity in [wealth distribution](/assets/edstudy/output_86_1.png).)

{% include gallery caption="Gallery of data visualizations for this project: (1) Dropout percentage by ethnicity, (2) Dropout rate by local median income, and (3) Dropout by school type." %}

The figures in the gallery are a sampling of the project. The first figure shows dropout percentage by ethnicity. Statewide, the dropout percentage for all students is ~1.5%. However, minorities such as African-Americans, Hispanics, and Native American and Alaska Natives are dropping out at a rate higher than this. Interestingly, the students who chose not to report their ethnicity showed a far higher rate of dropping out than any other category. However, when looking at [total enrollment](/assets/edstudy/output_37_1.png), the "Not reported" category of students is very small relative to the others. The second figure highlights dropout rate percentage by local school median income. Schools located in the poorest areas are the most at-risk for having a high dropout percentage. The dataset also allowed for looking at dropouts by school type, which is something that I had not initially thought to investigate. The vast majority (94.2%) of CA high school students are enrolled in public high schools, of which a very high proportion show very low dropout rates. On the other hand, juvenile court schools and county community schools show very high dropout rates. (I just happened to catch up with a high school friend recently who teaches at a juvenile court school. He gave me some insight into some of the challenges there.)

Importantly, there are many school factors that counteract apparent trends in dropout rates. What is particularly encouraging is the high number of schools that seem "at-risk" yet have low dropout rates. For example, in the figure showing dropout rate by local median income, half of the schools that are in the poorest areas of California still have dropout rates in the low single digits. One such school is Stockton Collegiate International Secondary School, which resides in a zip code with a local median income of $14,579 and yet has only 0.42% of students dropping out. Not surprisingly, they have earned accolades in US News & World Report according to their [website](http://www.stocktoncollegiate.org). 

A basic mission of our school system is to see that all students graduate high school and avoid dropping out. Students who drop out face significant life challenges, including lower wages and higher incarceration rates. However, the factors that influence a school's dropout rate are numerous and complex. The purpose of this data science project was to explore and identify features that could contribute to high dropout rate and find schools that have low dropout rates. Studying schools like Stockton Collegiate and performing additional analysis about them could help reveal factors that facilitate student success in challenging circumstances.

The [project repo](https://github.com/benslack19/CA-ed-study) contains Jupyter notebooks and additional figures.
