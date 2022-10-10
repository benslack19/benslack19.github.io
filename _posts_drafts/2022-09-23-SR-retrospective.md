---
title: "How I joined the Statistical Rethinking cult"
---

[Statistical Rethinking by Richard McElreath](https://xcelab.net/rm/statistical-rethinking/) appears to have gained a cult following in the last few years. I'm a part of it and I'm hardly the first to have joined. The first version of Solomon Kurz's tidyverse translation starts out unabashed: ["This is a love letter"](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/).

 The strong affinity that others have for the book is no doubt due to Dr. McElreath's personality and his attitude towards science and statistics. From his lectures and his writing style, he makes material accessible rather than the high-and-mighty professor that (consciously or unconsciously) gatekeeps.

I was not exactly a novice when I first encountered the book. I've taken several courses in statistics, applied it in research work, and had worked as a data scientist. I nevertheless felt uneasy about my statistics knowledge and the decision points to be made for applying a statistical procedure. Dr. McElreath's book was the first I've come across that spoke directly to my personal discomfort on the matter. Seeing the response from others to his book demonstrated that I was not alone.

I did not initially read SR to improve my general knowledge of statistics. Instead, I sought to better understand a specific subject: mixed effects models. I read online that SR might be a good resource that could help explain it. But when I jumped ahead to the mixed effects chapter, I didn't quite grasp the concepts and felt like I had to go to the beginning. Dr. McElreath states in his preface that the book is meant to be read sequentially. Despite the unexpected time investment to go through the book and course, I’m happy that I did.

 If you're reading this, you are likely considering going through the course as well. I hope you'll find it well worth your time. I hope me sharing my experience will help you on your journey. I went through the book initially on my own and started to hit a wall in the fifth chapter. I found an online study group where I joined them in going through the rest of the book. That took about 6 months, going at our own pace. I went through the book a second time, serving as a study group co-leader, running in parallel to the Winter 2022 series which was 10 weeks.

This post should not necessarily be interpreted as a list of things you "should" do. But I hope by sharing my experience it might give you ideas on how to get through this wonderful, yet challenging, course.

# Pre-requisites
**Statistics and math.** The inclusion of the word "rethinking" in the book title is apt. I mention my exposure to statistics above so that you can also appreciate where my comments are coming from. In addition to courses, I had gone through this [Khan Academy course](https://www.khanacademy.org/math/statistics-probability). If you understand those topics, I think you're okay moving forward.
<br>
**Programming.**  A commonly asked question is, "What package or approach should I use for the course?" This really is a matter of personal preference as [the course page](https://xcelab.net/rm/statistical-rethinking/) lists several repos in R, Python, and Julia. I had used both R and Python in work projects before beginning the course. The path with the least resistance would be to use the book's code (base R and `rethinking` package). I chose to use `pymc` and, while it may have taken me more time, thinking about coding implementation forced me to consider the material more deeply.

# Setting up
**Video series and accompanying course repos**.  In theory, one can use only the book's second edition to learn. While there is no shortage of McElreath jokes in his text, the videos exhibit more of his wit. There are several video series one can choose from. I went through the Winter 2019 series (two years after it came out) and Winter 2022 series (as they were released). I would recommend choosing from one of these. (While you can find video series older than 2019, they are reflective of the book's first edition.) Your choice of video series will essentially come down to this: how much do you want the material to align with the second edition of the book versus how "updated" do you want the lectures to be? 

| Video series  | Associated repo  |  Comments |
| --- | --- |  --- | 
| [Winter 2019 videos](https://www.youtube.com/watch?v=4WVelCswXo4&list=PLDcUM9US4XdNM4Edgs7weiyIguLSToZRI)  | [Winter 2020 repo](https://github.com/rmcelreath/stat_rethinking_2020/tree/main/homework)  <br> (There is a 2019 repo, but I'd recommend using the more recent version.)| This is tightly aligned with the second edition of the book. Several code examples in the lecture are exactly the same as the book. He delivers in front of an audience where his personality shines. |
| [Winter 2022 videos](https://www.youtube.com/playlist?list=PLDcUM9US4XdMROZ57-OIRtIK0aOynbgZN) | [Winter 2022 repo](https://github.com/rmcelreath/stat_rethinking_2022)  |  This was created during the pandemic and without a live audience, but he has been aware that he now has a larger, global following. Videos have great production quality and he speaks more slowly to accomodate students whose primary language is not English. In addition, it appears that he is iterating and testing both content and problems in preparation for a 3rd edition of the book. |

Both video series have their advantages and disadvantages. Personally, I liked the consistency between book, text, and repos that the Winter 2019 series offers. On the other hand, he is clearly working on improving his (already great) pedagogical methods. If you would rather use the most updated content that's in the Winter 2022 series, you won't have a problem using the second edition. But just be aware that there will be some inconsistency across resources. For example, the `pymc` version of the SR repo does not yet have the Winter 2022 lecture code as of the time this post was written (September 2022).

**Resources** 
<br>
Stuff to bookmark on your browser:
- [The main book page](https://xcelab.net/rm/statistical-rethinking/)
- Whatever video series and associated repo you choose above.
- I also liked to occasionally watch videos on my phone's YouTube app. Consider bookmarking and subscribing to [Dr. McElreath's YouTube channel](https://www.youtube.com/channel/UCNJK6_DZvcMqNSzQdEkzvzA).


**Books**

I like to have hard copies of books I reference a lot. You can get it [here](https://www.routledge.com/Statistical-Rethinking-A-Bayesian-Course-with-Examples-in-R-and-STAN/McElreath/p/book/9780367139919). If finances are an issue, look at your local library. If you're associated with a university, you may be able to get access to an electronic copy as well. I used both the hard and electronic copy.
<br>

**Social media sources**
- Follow Dr. McElreath and his network on Twitter (based on their auto-suggest). They'll post stats content which may not make sense to you immediately. As you progress through the course, it's a good sign when you start to recognize their memes and jokes. (There's great cat content too.)
- If you're using a package, you can follow their social media platforms. (The `pymc` folks are pretty active.)

**Getting through the course**
<br>
I had been reading the book [Atomic Habits](https://jamesclear.com/atomic-habits) around the time that I went through my first pass through the course. It helped me to go through SR due to some of the recommended practices. I know that this time commitment may not be realistic for everyone. Nevertheless, there were two things that I felt helped me: being consistent and not putting pressure on myself to learn everything on the first pass.

- I established a routine where I devoted 1-2 hours per day every morning. I'd print out the slides of the video series and write notes (two slides per page and double-spaced otherwise I couldn't staple).
- My order of exposure would generally go something like this: First watch the lecture and take notes. Then it would be a back and forth between the book's text and the homework problems (which were usually the book's hard problems in the 2020 course).
- I avoided using the answer key in the homework. I liked having a course repo to check my answers but, as much as possible, I tried to resolve my own confusions in individual or group study sessions. I felt like the lessons stuck that way. Dr. McElreath implores us to ["draw the owl"](https://glebbahmutov.com/blog/how-to-draw-an-owl/) throughout the course. 
- Having a group session was valuable. We met on a weekly basis and it helped me keep moving forward. Sometimes you'll get behind and sometimes you'll want to move ahead, but there's undoubtedly great benefit in both learning from and helping others. The Discord channel I was a part of also had other members who knew SR material well and had prior exposure to other statistics resources.
- I felt like the earlier chapters weren't too hard but the difficulty ramped up around Chapter 5 and 6. Having a routine established helps push through the harder sections.
- As material gets hard, you may find yourself getting frustrated or feel like you're taking too long to get through the course. Here's an important thing that helped me: whether you're in a group or studying by yourself, **decide on a criteria that determines whether you are okay moving on.** This speaks to the idea to not feel like you have to learn everything on the first pass. My criteria was that I could do at least one hard end-of-chapter problem (often it would be a homework problem). For others, it might be doing all of the homework. The Winter 2022 series had bonus problems but I had some trouble with them in the later weeks.
- Understand what you want to get out of the course and try to not get too lost in details. I wanted to learn generalized linear models and multilevel models. That meant I did not care to be an expert on Metropolis Monte Carlo. I just wanted to know a little bit about it to understand what it's doing. As another example, I found ordererd categorical models interesting but it wasn't my highest priority. It's helpful to know what you personally can de-prioritize.
- I felt most comfortable about a concept when I could apply an SR lesson to my own problem and write a blog post about it.

# After finishing the first time
Your priorities may vary after you've gone through the course. You may seek to:
- Revisit past concepts, including ideas you felt were lower priority initially
- Look at concepts from different sources. (This is what I aimed to do with various levels of success.) Here are a few books worth considering.



including The Book of Why, Bayesian Data Analyais, Gelman and Hill, Causal Inference: The Mixtape, etc.

# Imperfections
- Some errors?
- Many datasets are small, great teaching examples, not super realistic
- Culture change with others (e.g. mixed effects model)
- T-test/ANOVA teaser
- Open source contributions


# Closing thoughts
other reviews
- https://www.crosstab.io/articles/statistical-rethinking-review


https://www.tandfonline.com/doi/full/10.1080/10691898.2020.1806761

https://xianblog.wordpress.com/2016/04/06/statistical-rethinking-book-review/

https://statmodeling.stat.columbia.edu/2016/01/15/mcelreaths-statistial-rethinking-a-bayesian-course-with-examples-in-r-and-stan/

https://www.amazon.com/review/RTVBJSAQ4RKRM/ref=cm_cr_dp_title?ie=UTF8&ASIN=1482253445&channel=detail-glance&nodeID=283155&store=books




I revisited a section of statistics text recently and getting slightly teary eyed because I could comprehend stuff I wouldn’t have before SR. I hope it will make a similar impact on you. The material is not easy and throughout his videos he reminds us to "Be kind and patient with ourselves" and "If you're slightly confused, it only means you're paying attention." You should expect that the material will be difficult but I hope you will feel a sense of accomplishment when you complete it.