# Load Time-Series Analysis: Interpretation of the Plots

This document explains how to interpret the plots generated from the electricity load dataset and what they indicate for this specific dataset.

## Plot A: Entire Series

This figure shows the full five-year hourly electricity load series. It is useful for identifying the overall trend, long-term variability, and irregular events.

For this dataset, the load values fluctuate strongly over time and show a clear pattern of repeated cycles. The red rolling mean helps suppress the hourly noise and reveal the long-term movement of the series. The overall shape suggests a strong temporal structure, with demand rising and falling within daily and seasonal cycles. The figure also helps to identify large deviations, sudden jumps, or unusually flat or irregular segments that may correspond to holidays, outages, weather anomalies, or data quality issues.

Key interpretation:
- The series is not random; it has repeated temporal structure.
- High-frequency hourly variation is superimposed on a slower seasonal/trend component.
- Sudden changes or outliers should be checked to determine if they are real demand changes or anomalous data.

## Plot B: One Week

This figure zooms into a short time window to examine the daily operating cycle more clearly. It is useful for seeing the morning ramp-up, evening peak, and overnight low.

In this dataset, the weekly pattern shows a typical load profile: low demand during the night, an increase in the morning, a peak in the late afternoon or evening, and then a drop again at night. This pattern is consistent with a load curve driven by human activity, industrial usage, and residential electricity consumption. The figure also makes it easier to compare weekdays and weekends. In general, weekdays tend to have stronger and more structured peaks, while weekends often show reduced or shifted demand patterns.

Key interpretation:
- The daily cycle is strong and repeatable.
- The evening peak is the dominant demand period.
- Weekday and weekend profiles are visually different, indicating a weekly rhythm.

## Plot C: One Month

This plot focuses on a monthly window to assess whether the weekly rhythm repeats in a stable way across the month. It is useful for identifying whether the system behaves consistently from week to week and whether some weeks are unusual.

For this dataset, the monthly view suggests recurring weekly cycles rather than random fluctuations. The repeated rises and falls indicate a stable electricity demand pattern. Some weeks may appear slightly higher or lower than others, which may reflect weather, holidays, weekends, or exceptional events. Comparing weeks helps identify whether a particular period is unusually different from the usual pattern.

Key interpretation:
- Weekly repetition is visible in the monthly view.
- Some days or weeks may deviate from the norm, suggesting special conditions.
- A stable, repeating weekly rhythm is a sign of strong regularity in demand.

## Plot D: Daily Average and Hourly Average

This figure contains two different types of summaries.

### D1: Daily Average Over Time
This gives the average load for each calendar day. It smooths out hourly noise and makes the long-term behavior easier to see.

For this dataset, the daily average reveals the seasonal pattern and overall movement in electricity demand over time. It is useful for identifying whether demand increases or decreases over months and seasons. Daily averages also highlight periods where demand is unusually high or low.

### D2: Average by Hour of Day
This summarizes the load by hour across the full dataset and reveals the typical 24-hour profile.

For this dataset, the average by hour shows the strongest load around late afternoon/evening, with the lowest values during the early hours of the morning. This indicates a clear daily demand cycle. This type of plot is especially useful for identifying the time of the peak and the timing of the minimum load.

Key interpretation:
- The daily cycle is highly regular.
- Evening demand is notably higher than nighttime demand.
- Daily average trends help reveal seasonal and long-term behavior.

## Plot E: Weekday × Hour Heatmap

This heatmap shows the mean load for each hour of the day and each day of the week. It is one of the clearest ways to visualize weekday and weekend behavior together.

For this dataset, the heatmap shows that load is generally higher during daytime and early evening hours, with lower demand at night. It also suggests a strong difference between weekdays and weekends. The load pattern is not only higher on weekdays; the shape of the profile also changes. This often happens because work schedules, household activity, and commercial energy use differ across the week.

Key interpretation:
- Hour-of-day strongly influences demand.
- Day-of-week modulates the pattern.
- Weekday and weekend profiles differ in both magnitude and shape.

## Overall Conclusion

Taken together, the plots indicate that the electricity load data has a strong and interpretable structure. The dominant patterns are:
- daily demand cycles,
- weekly demand variation,
- clear evening peaks,
- lower overnight demand,
- seasonal or long-term changes over the five-year period.

These patterns are typical of real-world electricity consumption and are consistent with the idea that load depends strongly on time of day, day of week, and broader seasonal conditions.
