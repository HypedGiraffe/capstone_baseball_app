# Capstone Project - Baseball Modeling and Forecasting Application

## Problem Statement
This project was driven by a focus to create an interactive user experience with Baseball Player metrics. Using pybaseball, I obtained aggregate career and season-by-season data for baseball players since 2007.


The application offers two main features, the ability to make predictions and forecast specifically for individual players based on variables selected by the user.


For the project to be a success, the modeling and forecasting has to be deeply customizable for the user, while still being informative. The user will be able to experiment with an array of different features, and learn about how baseball player statistics influence one another. 

### Guide of This Repository

Code - Here you will find every notebook used to create the final version of the baseball prediction app.
  1. - Aquiring Pitcher Data - Used to get season by season data and career aggregate data for pitchers
  2. - Aquiring Batter Data  - Used to get season by season data and career aggregate data for batters
  3. - EDA - Plotting and examining example variables from pitching data
  4. - Individual Career Data - Used to aquire individual dataframes for pitchers and batters, season by season
  5. - Streamlit - File used to run the App, carrying out modeling and forecasting
  6. - requirements.txt - Used to fix dependencies for public app
 
Data - Career aggregate data for batters and pitchers, used for modeling and predicting

Individ_Data - Season by season data for indvidual pitchers and batters, used for forecasting

Deliverables - An executive summary of the project, as well as a technical analysis

Slides - Slides that were used to present the product

## Useful Links

Pybaseball(Where Data Was collected) - https://github.com/jldbc/pybaseball

Created Application - https://hypedgiraffe-capstone-baseball-app-codestreamlit-xq2zea.streamlit.app/

Note! - Must have metrics selected for prediction/modeling part in order to access forecasting section
      - In addition, some players do not work for some metric combinations, I believe due to lack of data due to injury. 

  
