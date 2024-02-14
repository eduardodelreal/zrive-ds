

## Context & Goal

### Overview
- Our customers use the company app to purchase items from our grocery store.
- Occasionally, we aim to promote specific items. This could be due to discontinuation plans, nearing expiration, or market share expansion goals.

### Challenges
- Push notifications are used to boost sales and encourage user engagement with these targeted products.
- However, excessive push notifications can lead to user dissatisfaction and app uninstalls, incurring significant costs.
- Current open rate for push notifications is about 5%.

### Objective
- Develop a product based on a predictive model.
- This model will identify users most likely to be interested in promoted items.
- Target these users with push notifications to increase effectiveness and reduce potential user annoyance.

## Requirements

### User Segmentation
- Focus on users who purchase the promoted item along with at least 4 other items (minimum 5-item basket).
- This is due to the high shipping costs for smaller orders which might exceed the gross margin.

### System Functionality
- Sales operators should be able to select an item via a dropdown or search bar.
- The system will identify the user segment to target.
- Operators can then trigger customizable push notifications to these users.

## Planning

### Timeline
- This tool is a high priority, given the competitive market dynamics.
- A Proof of Concept (PoC) is expected within one week.
- The goal is to go live within 2 to 3 weeks.

## Impact

### Expected Outcomes
- Module 3: TRD 2
- Target to increase monthly sales by 2%.
- Aim for a 25% boost in sales of selected items.
- Further details can be found in the sales department's push analysis report.




# Milestone 1: Exploration Phase

## Overview

In this phase, we will focus on building the predictive model with a clear understanding of our data. Our approach involves two key steps:

### Data Filtering
- **Objective**: Refine the dataset to include only orders with 5 items or more.
- **Rationale**: This aligns with our requirement to focus on users who make larger purchases, optimizing for shipping costs and gross margin.

### Model Building and Evaluation
- **Approach**: Utilize linear models for their simplicity and speed, as we aim for a Proof of Concept (PoC) in a short timeframe.
- **Process**:
  - Implement a train/validation/test split to evaluate the models.
  - Focus on linear models due to time constraints and their proven efficiency in similar scenarios.

### Expected Outcome
- A comprehensive report, notebook, or documentation detailing:
  - The performance and findings of the implemented models.
  - Insights on what worked well and what did not, including analysis of why certain approaches were more effective.
- Most importantly, the selection of a final model to proceed to Milestone 2.

