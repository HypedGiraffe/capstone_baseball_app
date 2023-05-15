import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#Title for application
st.title("Baseball Modeling and Forecasting App")


#First Question for the User, Pitcher or Batter?
#Will Divert the Dataset for the rest of the code
p_type = st.sidebar.selectbox("Select Player Type", ("Pitcher", "Batter"))



#Questions for the user
#Will Provide the user with options for both variables to use in the modeling, as well as what reponse variable they are interested in
if p_type == "Batter":
    metrics = st.sidebar.multiselect("Select Metrics to Predict With", ("AVG", "SLG", "Seasons", "OPS", "Age", "WAR", "OBP", "wOBA", "BABIP", "HR", "RBI", "Contact%"))

if p_type == "Pitcher":
    metrics = st.sidebar.multiselect("Select Metrics to Predict With", ("Seasons", "H", "WAR", "ERA", "K/9", "H/9", "ER", "HBP", "K-BB%", "Contact%", "FB%"))
    
if p_type == "Batter":
    response = st.sidebar.selectbox("Select Metric to Predict", ("AVG", "SLG", "Seasons", "OPS", "Age", "WAR", "OBP", "wOBA", "BABIP", "HR", "RBI", "Contact%"))

if p_type == "Pitcher":
    response = st.sidebar.selectbox("Select Metric to Predict", ("Seasons", "H", "WAR", "ERA", "K/9", "H/9", "ER", "HBP", "K-BB%", "Contact%", "FB%"))

    
#Reading in relevant data, batter or pitching career data based on user input
df_career = pd.read_csv(f"/main/Data/career_{p_type.lower()}.csv")

#Intializes data frames for X and y for linear regression
X = pd.DataFrame()
y = pd.DataFrame()

#Adding relevant columns to our X and Y for regression
for i in metrics:
    X[i] = df_career[i]
y[response] = df_career[response]


#Train test/splitting our two data frames
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size = .2,
    random_state = 34
)

#Pipepine for Linear Regression - Will scale data
pipe = Pipeline(
    steps = [
        ('poly', PolynomialFeatures()),
        ('ss', StandardScaler()),
        ('lr', LinearRegression())
    ]
)

#Fits our model, creates predictions, and finally prints a mean squared error and R^2 for the user
#to evaluate their model
pipe.fit(X_train, y_train)                                                                          
preds = pipe.predict(X_train)                                                                          
metric = mean_squared_error(preds, y_train)
r2 = pipe.score(X_val, y_val)

st.write(""" Some Metrics for Your Model! -

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
""")
st.write(f"""
Your model obtained a mean squared error of {metric}

""")
st.write(f"""
Your model obtained a r square value of {r2}
""")

predictions = []
for i in range(0, len(preds)-1):
    predictions.append(preds[i])

output_df = pd.DataFrame()
output_df[response] = predictions

#Out puts a data frame of predictions
st.dataframe(output_df)



st.write(f"""

# Moving On To Forecasting
""")


###
###
###
#Forecasting Component

#Asking the user to pick which player they would like to forecast for
if p_type == 'Pitcher':
    player = st.sidebar.selectbox("Select Player To Forecast",
('Justin Verlander',
 'Clayton Kershaw',
 'Max Scherzer',
 'Zack Greinke',
 'Adam Wainwright',
 'David Price',
 'Corey Kluber',
 'Gerrit Cole',
 'Madison Bumgarner',
 'Jose Quintana',
 'Johnny Cueto',
 'Yu Darvish',
 'Lance Lynn',
 'Anibal Sanchez',
 'Carlos Carrasco',
 'Charlie Morton',
 'Kevin Gausman',
 'Sonny Gray',
 'Patrick Corbin',
 'Kenley Jansen',
 'Chris Archer',
 'Dallas Keuchel',
 'Craig Kimbrel',
 'Aroldis Chapman',
 'Nathan Eovaldi',
 'Wade Miley',
 'Rich Hill',
 'Kyle Gibson',
 'Alex Wood',
 'Mike Minor',
 'Alex Cobb',
 'Jhoulys Chacin',
 'Jake Odorizzi',
 'Martin Perez',
 'Collin McHugh',
 'Ian Kennedy',
 'David Robertson',
 'Liam Hendriks',
 'Michael Wacha',
 'Garrett Richards',
 'Mark Melancon',
 'Daniel Hudson',
 'Sergio Romo',
 'Ryan Pressly',
 'Matt Moore',
 'Steve Cishek',
 "Darren O'Day",
 'Adam Ottavino',
 'Craig Stammen',
 'Jake McGee',
 'Brad Hand',
 'Tommy Hunter',
 'Jordan Lyles',
 'Jesse Chavez',
 'Tommy Milone',
 'Joe Smith',
 'David Phelps',
 'Will Smith',
 'Alex Colome',
 'Joe Kelly',
 'Jeurys Familia',
 'Aaron Loup',
 'Jake Diekman',
 'Wily Peralta',
 'Erasmo Ramirez',
 'Bryan Shaw',
 'Ross Detwiler',
 'Anthony Bass',
 'Brad Boxberger',
 'Javy Guerra',
 'T.J. McFarland',
 'Javy Guerra'))

if p_type == 'Batter':
    player = st.sidebar.selectbox("Select Player to Forecast", ('Mike Trout',
 'Joey Votto',
 'Robinson Cano',
 'Evan Longoria',
 'Yadier Molina',
 'Miguel Cabrera',
 'Paul Goldschmidt',
 'Andrew McCutchen',
 'Freddie Freeman',
 'Jose Altuve',
 'Manny Machado',
 'Josh Donaldson',
 'Nolan Arenado',
 'Bryce Harper',
 'Albert Pujols',
 'Giancarlo Stanton',
 'Jose Ramirez',
 'Nelson Cruz',
 'Yasmani Grandal',
 'Christian Yelich',
 'Elvis Andrus',
 'Anthony Rizzo',
 'Justin Turner',
 'Justin Upton',
 'Starling Marte',
 'Xander Bogaerts',
 'Anthony Rendon',
 'Jason Heyward',
 'Matt Carpenter',
 'Lorenzo Cain',
 'Brandon Crawford',
 'Michael Brantley',
 'J.D. Martinez',
 'DJ LeMahieu',
 'Marcus Semien',
 'Jean Segura',
 'Brandon Belt',
 'Andrelton Simmons',
 'Carlos Santana',
 'Marcell Ozuna',
 'A.J. Pollock',
 'Jed Lowrie',
 'Kolten Wong',
 'Charlie Blackmon',
 'Mike Zunino',
 'Jonathan Schoop',
 'Eduardo Escobar',
 'Yan Gomes',
 "Travis d'Arnaud",
 'Salvador Perez',
 'Mike Moustakas',
 'Wil Myers',
 'Jason Castro',
 'Kole Calhoun',
 'Jackie Bradley Jr.',
 'Josh Harrison',
 'Cesar Hernandez',
 'Jose Iglesias',
 'Martin Maldonado',
 'Alcides Escobar',
 'Corey Dickerson',
 'Nick Castellanos',
 'Didi Gregorius',
 'Aaron Hicks',
 'Billy Hamilton',
 'Marwin Gonzalez',
 'Kevin Pillar',
 'Dee Strange-Gordon',
 'Avisail Garcia',
 'Wilmer Flores',
 'Eric Hosmer',
 'Jonathan Villar',
 'Kurt Suzuki',
 'Jake Marisnick',
 'Josh Bell',
 'Robbie Grossman',
 'Brad Miller',
 'Robinson Chirinos',
 'Stephen Vogt',
 'Ehire Adrianza',
 'Leury Garcia',
 'Abraham Almonte',
 'Sandy Leon',
 'Chris Owings',
 'Austin Romine',
 'Josh Bell'))

#Reading in player data based on their input
df_player = pd.read_csv(f"../Individ_Data/{p_type}/{player}.csv")
df_player.set_index('Date', inplace = True)

#Prints a dataframe of the players career data, season by season
st.write(f"""
A look at {player}'s statistics by season
""")
st.write("""
""")
st.dataframe(df_player)

#Now, we want user input of which metrics to use
if p_type == "Pitcher":
    metrics_ts = st.sidebar.multiselect(f"Select Metrics to Forecast for {player}", ("W", "L", "G", "WAR", "ERA", "K/9", "BB", "BABIP"), max_selections=3)

if p_type == "Batter":
    metrics_ts = st.sidebar.multiselect(f"Select Metrics to Forecast for {player}", ("Age", "AB", "H", "G", "WAR", "RBI", "SO", "BB", "BABIP", "GB%", "FB%"))
    
    
#Plotting Time Series
# Function called plot_series that takes in 
# a dataframe, a list of column names to plot, the 
# plot title and the axis labels as arguments,
# then displays the line plot with a figure size
# of 18 horizontal inches by 9 vertical inches.

# Matthew Garton - BOS, taken from DSI-GA Time Series Lecture

def plot_series(df, cols=None, title='Title', xlab=None, ylab=None):
    
    # Set figure size to be (18, 9).
    plt.figure(figsize=(18,9))
    
    # Iterate through each column name.
    for col in cols:
        
        # Generate a line plot of the column name.
        # You only have to specify Y, since our
        # index will be a datetime index.
        plt.plot(df[col])
        
    # Generate title and labels.
    plt.title(title, fontsize=26)
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    
    # Enlarge tick marks.
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18);

    st.set_option('deprecation.showPyplotGlobalUse', False)

#Creates a time series plot for each metric the user created
for i in metrics_ts:
    st.pyplot(plot_series(df_player, cols=[i], title = f"{player}'s {i} By Year", xlab = 'Year', ylab = i))

# Code written by Joseph Nelson, from GA VAR lecture
#Tests for stationarity
def interpret_dftest(dftest):
    dfoutput = pd.Series(dftest[0:2], index=['Test Statistic','p-value'])
    return dfoutput

#Gets all of the interested variables for our user, into one list
col_ts = []
for i in metrics_ts:
    col_ts.append(df_player[i])

    
times_diff = []
#Iterates over the metric columns, and gets them stationary, also keeps track of how many times the data
#was differenced
for i in range(0, len(col_ts)):
    curr = 0
    while interpret_dftest(adfuller(col_ts[i].dropna()))[1] > .05:
        col_ts[i] = col_ts[i].diff()
        curr += 1
    times_diff.append(curr)
        
#Creates a data frame called stationary_df, this is the stationary version of our data
stationary_df = pd.DataFrame()
for i in range(0, len(col_ts)):
    stationary_df[metrics_ts[i]] = col_ts[i]

#Gets rid of years will nulls
stationary_df.dropna(inplace = True)

#Train test splitting our data
train, test = train_test_split(stationary_df,
                               test_size = 0.10, shuffle=False)

#Fits a var model to our data, and also creates forecasted values
model = VAR(train)
ts_model = model.fit(maxlags=1, 
                     ic = 'aic')   
lag_vals = train.values[-2:]
pre = ts_model.forecast(y=lag_vals, steps=1)


#Looked at https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/
#Helped me figure out how to return data to pre-differenced



#Finally, prints the forecasted data to the user
st.write("""

""")

st.write("""
# Forecasts for 2023 Season!
""")


st.write(f"""
{player} is predicted to have {metrics_ts[0]} = {pre[0][0]} in the 2023 season
""")

st.write(f"""
{player} is predicted to have {metrics_ts[1]} = {pre[0][1]} in the 2023 season
""")

st.write(f"""
{player} is predicted to have {metrics_ts[2]} = {pre[0][2]} in the 2023 season
""")
