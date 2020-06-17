---
title: The probability of making your Friday night party
toc: true
categories: [data science, statistics]
---

My wife and I have enjoyed living in the Bay Area where we've been able to satisfy our love of outdoor activities while being near a cool city. While we're fortunate to have friends and family in the area, I've felt bad that we haven't been able to hang out with them as often as we'd like. We knew that part of this was due to our work schedules and occasional travel. I work a regular M-F schedule, but my wife has a very irregular schedule as a nurse. She "only" works three days a week, but those are 12-hour shifts and she's required to work every other weekend. This has limited us going from attending [trivia](https://www.yieldandpause.com/trivia/), [sporting events](https://www.nhl.com/sharks), or just your [average party](https://media.giphy.com/media/wAxlCmeX1ri1y/giphy.gif). In addition, since both of our immediate families are not in the area, we might be hosting or visiting on weekends that we both have off. While everyone is busy and has commitments, I wanted to quantify how much we could get together socially with our friends in the area. Therefore, in this post, I explore the question: **Given a social time, what is the probability that we're able to get together?**

## Parameters for the social schedule

1. My wife works 3 days in a work week, but not more than than 3 days in a row, when considering consecutive weeks.
2. She works every other weekend on both Saturday and Sunday.
3. Hanging out on weekday evenings means she has to be off on the day of and the day after since she has to get up early for work.

Possible social times can be any evening get-together. I'm being pretty generous since many people do not meet during the week. But trivia is on Tuesdays and others might propose a random happy hour here or there so I think evening get-togethers are fair game. Day time get-togethers are also possible on Saturday and Sunday. On weekends, we could theoretically see some people during the day (a brunch or hike) and others in the evening (dinner or party). That gives us 8 possible time frames in a given week.


```python
# Import packages
import pandas as pd
import numpy as np
import random
```

## Create a social schedule for the year


```python
days_of_week = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
```


```python
# I'll look over the course of a year, giving me 52 weeks to simulate.
no_weeks = 52

# Week will start on a Monday. A work day is indicated by a 1.
schedule_by_week = pd.DataFrame(np.zeros((no_weeks, 7)), columns=days_of_week)

workweek_params_wkend_off = np.asarray([0,0,1,1,1])  # 3 work days M-F on weekends she has off
workweek_params_wkend_on = np.asarray([0,0,0,0,1]) # 1 work days M-F on weekends she has on

# Set the first week of the year.
schedule_by_week.loc[0, :] = np.concatenate((workweek_params_wkend_off, np.asarray([0,0])))

for i in range(1, no_weeks):
    # This block accounts for working every other weekend and not working >3 days in a row
    
    # The if statement prevents Friday, Monday around a working weekend
    if (i > 0) & (i % 2 == 0) & (schedule_by_week.loc[(i-1), 'Fri']==1):
        np.random.shuffle(workweek_params_wkend_off)  # shuffles in place
        while workweek_params_wkend_off[0]==1:
            np.random.shuffle(workweek_params_wkend_off) 
        schedule_by_week.loc[i, :] = np.concatenate((workweek_params_wkend_off, np.asarray([0,0])))
    
    # The elif statement prevents working both Monday and Tuesday following a working weekend
    elif i % 2 == 0:
        np.random.shuffle(workweek_params_wkend_off) 
        while workweek_params_wkend_off[0]==1 & workweek_params_wkend_off[1]==1:
            np.random.shuffle(workweek_params_wkend_off) 
        schedule_by_week.loc[i, :] = np.concatenate((workweek_params_wkend_off, np.asarray([0,0])))
    
     # The else statement randomly assigns a M-F day to work when working on the weekend
    else: 
        np.random.shuffle(workweek_params_wkend_on)
        schedule_by_week.loc[i, :] = np.concatenate((workweek_params_wkend_on, np.asarray([1,1])))
```


```python
# View the first few weeks of the simulated year as a sanity check
schedule_by_week.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mon</th>
      <th>Tues</th>
      <th>Wed</th>
      <th>Thur</th>
      <th>Fri</th>
      <th>Sat</th>
      <th>Sun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Make the social schedule a little more realistic

At this point, I have a simulated week every week of the year and looking at the data at this point seems to match the conditions I listed above. Before calculating the probabilities of get-togethers, I'm going to make some modifications to make it more realistic for my question:
1. I'll remove the first and last weeks of the year, since both our friends and us likely won't be available over the holidays.
2. I'll eliminate two full weeks from our calendar in the middle of the year for vacation. This doesn't affect our friends' schedule, so it's a possibility that we would miss something that they host (like a 4th of July BBQ). Therefore, this won't affect the denominator of the probability in the calculation.
3. I'll remove 8 weekends from consideration for us taking weekend trips, hosting guests from out of town, doing weekend-long home  projects, or just chilling by ourselves. Like modification #2, this would not affect the denominator. Eight was somewhat arbitrary but thought this was a reasonable number.


```python
# Modification 1, drop the first and last weeks
schedule_by_week.drop(labels=[0, 51], inplace=True)
```


```python
# The denominator for each social time frame
count_friends_socialweeks = schedule_by_week.shape[0]
print('Total possible weeks for socializing: ', count_friends_socialweeks)
```

    Total possible weeks for socializing:  50



```python
# Factor in modifications 2 and 3 
```


```python
# Account for two weeks we would be on vacation in the summer (modification 2)
schedule_by_week2 = schedule_by_week.drop(labels=[24, 25]) 
```


```python
# Remove 8 weekends from social consideration (modification 3).
# I'll simply change the even-indexed weekends to all 1's. Since these aren't work days per se,
# I won't worry about the "no more than 3 work days in a row rule" that I listed at the beginning.
mask1 = range(max(schedule_by_week2.index))
mask2 = np.intersect1d(mask2, schedule_by_week2.index)   # Account for weeks I dropped in mods 1 and 2

# Randomly choose 8 weeks where the weekends will be changed to 1's
mask3 = np.random.choice(mask2, size=8, replace=False)

schedule_by_week2.loc[mask3,['Sat', 'Sun']] = 1
```


```python
print('Weeks with weekends affected by modification 3: ', mask3)
```

    Weeks with weekends affected by modification 3:  [ 4 18  2 34 12 20 26 42]



```python
schedule_by_week2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mon</th>
      <th>Tues</th>
      <th>Wed</th>
      <th>Thur</th>
      <th>Fri</th>
      <th>Sat</th>
      <th>Sun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Assess probabilities for a possible social window


```python
# Test case: probability of meeting on a Monday night

# Meeting on a Monday evening requires that she's off Monday and Tuesday.
mask = (schedule_by_week2.loc[:,'Mon']==0) & (schedule_by_week2.loc[:,'Tues']==0)
```


```python
# Sanity check: the first 5 weeks that she's off both Monday and Tuesday
schedule_by_week2[mask].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mon</th>
      <th>Tues</th>
      <th>Wed</th>
      <th>Thur</th>
      <th>Fri</th>
      <th>Sat</th>
      <th>Sun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Probability of meeting on a Monday night
schedule_by_week2[mask].shape[0]/count_friends_socialweeks
```




    0.26



### Assessing probabilities of meeting on weekday evenings.


```python
for i, day in enumerate(days_of_week[:5]):
    day = days_of_week[i]
    nextday = days_of_week[i+1]
    # Meeting on a given evening requires that Kathleen is off that and the following evening.
    mask = (schedule_by_week2.loc[:,day]==0) & (schedule_by_week2.loc[:,nextday]==0)
    print(day, 'evening: ', int(100*(schedule_by_week2[mask].shape[0]/count_friends_socialweeks)), '%')
```

    Mon evening:  26 %
    Tues evening:  32 %
    Wed evening:  34 %
    Thur evening:  30 %
    Fri evening:  12 %



```python
# The probability of getting together day time or evening on Saturday are the same since being off on a Saturday means also being off on a Sunday.
mask = (schedule_by_week2.loc[:,'Sat']==0) # & (schedule_by_week2.loc[:,'Sun']==0)
print('Sat day or evening: ', int(100*(schedule_by_week2[mask].shape[0]/count_friends_socialweeks)), '%')

# For Sunday, I'm ignoring the possibility of her working on Monday since a Sunday get-together will likely end before 9 pm anyway.
mask = (schedule_by_week2.loc[:,'Sun']==0)
print('Sun day or evening: ', int(100*(schedule_by_week2[mask].shape[0]/count_friends_socialweeks)), '%')
```

    Sat day or evening:  32 %
    Sun day or evening:  32 %


## Summary

As you can see, it's clear that Friday evening is most affected by our schedules. Of course, that's a common time to get together for most. I wouldn't have necessarily guessed that it would be this much lower but it makes sense when thinking about it. On weeks where my wife is assigned to weekends, Friday is the only weekday that also decreases in probability of getting together. It's also good to point out that these probabilities are likely overestimates. I didn't account for sick days, errands, family emergencies, or other things that affect our time. From one perspective, we wish we could find more time to get together. But on the other hand, when we do manage to get together, we're grateful for the chance (literally) to do so.

This post is dated, but you could probably guess when I did most of the work even if you didn't look. It's a Friday night.
