
# Waiting Time Prediction
[![ch](https://img.shields.io/badge/lang-ch-green.svg)](https://github.com/I-Sheng/Traveling-Planner-With-LLM/blob/main/data/waiting_time/README.md)

## Content Overview
When planning a trip, one crucial factor to consider is the amount of time spent at each spot. However, data retrieved from the Google API does not always include waiting time. This project aims to use popular times data to train a model to predict the waiting time for a site, enabling better planning for travelers.

Considering the potential sparsity of data, I collected popular time data from all spots within Chiayi City, Chiayi County, and Tainan to build and improve the predictive model.

## Data Collection Steps
1. **Install Required Module**
   * Install the [Populartimes module](https://github.com/m-wrzr/populartimes) to retrieve popular time and waiting time data from the Google API.

2. **Gather Spot Names for Data Collection**
   * Use the following tourism sites to obtain the spot names:
     - [Travel in Chiayi City](https://travel.chiayi.gov.tw/)
     - [Culture & Tourism Bureau of Chiayi County](https://tbocc.cyhg.gov.tw/)
     - [TRAVEL TAINAN](https://www.twtainan.net/)

3. **Retrieve Popular Times and Waiting Time Data**
   * Use Python scripts to fetch data categorized into four types:
     1. **nopopular_notimeSpent.json**: Spots with no popular time and no waiting time data.
     2. **popular_notimeSpent.json**: Spots with only popular time data. The model aims to predict waiting time using this data.
     3. **nopopular_timeSpent.json**: Spots with only waiting time data.
     4. **popular_timeSpent.json**: Spots containing both popular time and waiting time data, which will be used to train and test the predictive models.

   * Example commands to fetch data:
     ```bash
     python3 getPopular_chiayi.py ./webcrab/chiayi_food.txt
     # For Chiayi, use the getPopular_chiayi.py script

     python3 getPopular_tainan.py ./webcrab/tainan_food.txt
     # For Tainan, use the getPopular_tainan.py script
     ```

## Model Training
* After data collection, train a model to predict waiting times using the popular times data.
* Details on the prediction process and model performance can be found in `prediciton_model`.

## Future Improvements
- Experiment with different machine learning models and feature engineering to improve waiting time predictions.
- Expand data collection to additional regions or spots to enhance model robustness.
- Integrate the model into a user-friendly tool or API for easy access during travel planning.
