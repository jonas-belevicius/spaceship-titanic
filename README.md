# Spaceship Titanic

## Description

In this competition task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help you make these predictions, you're given a set of personal records recovered from the ship's damaged computer system.

## File and Data Field Descriptions

**train.csv** - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.

* `PassengerId` - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
* `HomePlanet` - The planet the passenger departed from, typically their planet of permanent residence.
* `CryoSleep` - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
* `Cabin` - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
* `Destination` - The planet the passenger will be debarking to.
* `Age` - The age of the passenger.
* `VIP` - Whether the passenger has paid for special VIP service during the voyage.
* `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
* `Name` - The first and last names of the passenger.
* `Transported` - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

**test.csv** - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.

## Goal

* Make a model that predicts, whether a passenger was transported to an alternate dimension and to find the features that are the most important for this prediction

## Objectives

**Data Aquisition and Understanding**

* Analyse, how different passenger parameters are associated with transportation to another dimension.
* Analyse, how different passenger parameters are associated with each other.
* Analyse, how different passenger parameters are correlated.
* Identify groups of similar passengers.

**Modeling and Evaluation**

* Apply different ensemble methods to predict whether a passenger was transported to alternate dimension.
* Tune the models to find the best hyperaparameters.
* Identify the most important features for model predictions.

## Data Aquisition and Understanding Findings

**Analyse, how different passenger parameters are associated with transportation to another dimension.**

* Passengers that were more likely to be transported to another dimension:
  * Passengers from `Europa` or `Mars`.
  * Passengers in `Cryo sleep`.
  * Passengers travelling to `*55 Cancri e*`.
  * Passengers located in `cabin decks B` and `C` and in lower `cabin numbers`.
  * `Not-VIP` passengers.
  * Passengers spending less on services - `room service`, `food court`, `shopping mall`, `spa`, `VRDeck`.

**Analyse, how different passenger parameters are associated with each other.**

* `Cabin number` is lineary associated with passenger group. Most of the passengers
travelling in the same group are located in the same `cabin number` and `deck` and all passengers
from the same `group` are located in the same `cabin side`.

* Most of the passengers travelling to PSO J318.5-22 are from Earth. All passengers in decks
B, A, C and T are from Europe, and passengers in G deck are from Earth. Passengers in
other decks are from different `Home Planets`.

* Passengers who were in cryo sleep did not spend anything on services.

**Analyse, how different passenger parameters are correlated.**

* Passengers spending more on `shopping` also spend more on `room service`. Passengers who
spend more on `spa`, also spend more on `food court` and `VRDeck`.

**Identify groups of similar passengers.**

* Clustering identified ~20 different passenger clusters that describe ~1500 passengers.

## Conclusions

* Best modeling results were showed by CatBoostClassifiers and their tuned versions.
* Different optimization methods found different optimal classifiers. The best tuned classifiers were CatBoostClassifier, XGBClassifier and LGBMClassifier.
* Ensemble prediction results achieved overall accuracy of 0.8036 on test dataset.
