# DATA CLEANING
We have 43 data points with no total_purchase value. we drop these 43 orders.

# OUTLIERS
We have an extreme point which obviously is detected as an outlier.
This point is an order with total_purchase of 100,000,000.
Investigating this user's other purchases demonstrates that total amount and average of the 7 others purchases
are respectively 2,125,000 and (~)303,571. This registered order can be a normal order as well as a fraud or even a mistake in our system!
Although I can omit this order but I prefer to keep it since I don't have domain knowledge.
Furthermore the questioner has asked me to just report specific things and no preprocessing has been requested in the question!