## User Churn Project – Waze App
Client: Waze Leadership Team
Objective: Increase user retention and prevent monthly app churn through data-driven insights and predictive modeling.

##  Problem
User churn—defined as users uninstalling or ceasing to use the Waze app—was impacting overall growth. The goal was to identify churn patterns and create predictive models to help inform marketing and product strategies aimed at increasing retention.

##  Preliminary Data Summary (Milestone 2)
Dataset Overview:

12 unique variables (objects, floats, integers).

82% retained users vs 18% churned users.

Missing values detected in the label column (~700 cases).

Key Findings:

Churned users had more drives (~3), but used the app on fewer days.

On average, churned users drove 200 km and 2.5 more hours than retained users.

Churned users drove farther per driving day (~698 km/day), ~240% more than retained users.

Next Steps:

Investigate behavioral patterns of high-usage "super-drivers".

Begin full exploratory data analysis and visual storytelling.

##  Exploratory Data Analysis (Milestone 3)
Key Correlations:

##  Negative: App usage days ⟶ Less churn.

##  Positive: Distance per driving day ⟶ Higher churn.

Distributions:

Variables were largely right-skewed or uniformly distributed.

Anomalies:

Detected improbable outliers (e.g., unrealistic driven_km_drives).

Action Points:

Deeper statistical analysis to explore potential churn profiles.

Clean and validate erroneous data for improved modeling.

##  Hypothesis Testing (Milestone 4)
Test: Two-sample t-test on mean number of drives by device type (Android vs iPhone).

Result: No statistically significant difference between groups (iPhone: 68, Android: 66 average drives).

Recommendation: Conduct further tests on other features to refine understanding of churn drivers.

##  Regression Modeling (Milestone 5)
Model: Binomial Logistic Regression

Evaluation:

Precision: 53%

Recall (critical for churn detection): Only 9% – model missed most churners.

Top Feature: activity_days had strong negative correlation with churn.

Insight: High km_per_driving_day, although relevant in EDA, was a weak predictor in the model.

Recommendation: Gather richer feature data and reconsider the user profile definitions before acting on regression insights.

##  Machine Learning Modeling (Milestone 6)
Approach:

Compared Random Forest vs XGBoost.

Data split into train/validation/test sets for fair model evaluation.

Outcome:

Insufficient Predictive Performance: Both models underperformed, highlighting limitations of the current dataset.

Recommendations:

Collect more granular drive-level and interaction-level data.

Engineer behavioral features (e.g., hazard reporting frequency, origin/destination patterns).

Proceed with a second iteration using expanded and enriched datasets.

##  Tools & Techniques
Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost)

Logistic Regression, Random Forest, Hypothesis Testing, EDA

Data Cleaning, Feature Engineering, Model Evaluation Metrics (Precision, Recall, Accuracy)

##  Key Takeaways
Frequent usage and shorter drive distances are key predictors of retention.

Heavy users churn differently—possibly due to unmet high-usage needs.

Data quality and depth are critical; predictive models require more behavioral features to be effective.

The iterative process (EDA → Hypothesis Testing → Regression → ML) provides a strong framework for actionable user insights.
Conclusions – Waze User Churn Project
Throughout this project, we explored behavioral patterns, tested statistical relationships, and developed predictive models to better understand and anticipate monthly user churn in the Waze app. Here are the key takeaways:

Retention is strongly tied to user engagement:
Users who used the app more frequently—especially across more days in a month—were significantly less likely to churn. In contrast, users with fewer sessions but higher average driving distances were more likely to leave the platform.

"activity_days" is the strongest predictor:
Across all analyses and models, the number of active days per month showed the highest correlation with churn. This variable consistently had the greatest impact in both statistical tests and machine learning models.

Model performance was limited:
Neither the logistic regression nor the machine learning models (Random Forest and XGBoost) performed well at predicting churn. The logistic regression achieved acceptable precision but had very low recall—only correctly identifying 9% of actual churners.

Data limitations were a major barrier:
The lack of more relevant, detailed features significantly restricted the models’ predictive power. Future efforts should focus on collecting richer data such as:

Driving session-level details (time of day, locations, trip duration)

In-app behavior (usage of features, alert confirmations, engagement patterns)

A cleaner and more complete churn label (currently many missing values)

Outliers (super-drivers) may need separate analysis:
A small group of users displayed unusually high driving distances in very few days. These super-drivers may represent a different user segment with unique needs, suggesting a potential benefit from personalized retention strategies.

