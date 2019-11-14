import os
import pickle
import sqlite3

import pandas as pd
from pandas.core.reshape.util import cartesian_product


def get_demand(conn):
    """Hourly trips per station"""

    # Get all trips
    trips = pd.read_sql_query("select * from trip;", con=conn)
    station_info = pd.read_sql_query("select * from station;", con=conn)

    # Convert to date
    station_info['installation_date'] = pd.to_datetime(
        station_info['installation_date'])
    trips['start_date'] = pd.to_datetime(trips['start_date'])

    # Convert to date and hour
    # trips['date_hour'] = trips['start_date'].dt.floor('h')

    # Convert to date and every 3 hour
    trips['date_hour'] = trips['start_date'].dt.floor('3h')

    # Get all combinations of time and station
    date_hours, stations = cartesian_product([
        pd.date_range(trips['date_hour'].min(),
                      trips['date_hour'].max(), freq='3h'),
        trips['start_station_id'].unique()])

    all_periods = pd.DataFrame(
        {'date_hour': date_hours, 'start_station_id': stations})

    # Keep station hour combo if it exists at that time
    all_periods = all_periods.merge(
        station_info[['id', 'installation_date']],
        how='left',
        left_on='start_station_id',
        right_on='id'
    ).query('installation_date<date_hour')

    # Trips by station and hour
    hourly_trips = (
        all_periods.merge(
            trips.groupby(['start_station_id',
                           'date_hour'])
            .size()
            .reset_index(name='demand'),
            how='left',
            on=['date_hour', 'start_station_id'])
        .fillna(0)
    )

    # Add time features
    hourly_trips['month'] = hourly_trips['date_hour'].dt.month
    hourly_trips['weekday'] = hourly_trips['date_hour'].dt.weekday
    hourly_trips['hour'] = hourly_trips['date_hour'].dt.hour
    hourly_trips['date'] = hourly_trips['date_hour'].dt.floor('d')

    return hourly_trips


def get_station_info(conn):
    """Station specific information"""

    # Get all station info
    station_info = pd.read_sql_query("select * from station;", con=conn)
    station_info['installation_date'] = pd.to_datetime(
        station_info['installation_date'])

    # Get zip_code for station
    city_zip_mapping = pd.DataFrame(
        {
            'city': [
                'San Jose',
                'Redwood City',
                'Mountain View',
                'Palo Alto',
                'San Francisco'],
            'zip_code': [
                95113,
                94063,
                94041,
                94301,
                94107]})

    station_info = station_info.merge(city_zip_mapping)

    return station_info


def get_weather():
    """Get weather data"""

    weather = pd.read_csv('data/weather.csv')
    weather['date'] = pd.to_datetime(weather['date'])

    return weather


def get_censored_demand(conn):
    """Get hourly censored demand by station (zero out)"""

    censored_demand = pd.read_sql_query(
        """select distinct station_id as start_station_id,
                    substr(time,1,13) as date_hour
                        from status
                        where bikes_available = 0;""",
        con=conn)

    censored_demand['date_hour'] = pd.to_datetime(censored_demand['date_hour'])
    censored_demand['date_hour'] = censored_demand['date_hour'].dt.floor('3h')
    censored_demand = censored_demand.drop_duplicates('date_hour')
    censored_demand['censored'] = 1

    return censored_demand


def create_model_obs(conn, weather=False):
    demand = get_demand(conn)
    censored = get_censored_demand(conn)
    stations = get_station_info(conn)

    model_data = (demand.merge(censored,
                               on=['start_station_id', 'date_hour'],
                               how='left')
                  .fillna(0)
                  .merge(stations[['id', 'zip_code']]
                         .rename(columns={'id': 'start_station_id'}),
                         on='start_station_id'))

    if not weather:
        return model_data
    else:
        weather = get_weather()[['zip_code', 'date',
                                 'mean_temperature_f',
                                 'precipitation_inches']]

        weather = weather[weather[['mean_temperature_f',
                                   'precipitation_inches']]
                          .notnull().all(axis=1)]

        model_data = (model_data
                      .merge(weather,
                             on=['zip_code', 'date']))
    return model_data


if __name__ == '__main__':

    conn = sqlite3.connect('data/sf_bikeshare.sqlite')
    demand = create_model_obs(conn, weather=True)

    if not os.path.exists('data'):
        os.mkdir('data')

    train = demand.sample(frac=0.5, random_state=42)
    test = demand.drop(train.index)

    with open('data/final_train.pickle', 'wb') as f:
        pickle.dump(train, f)
    with open('data/final_test.pickle', 'wb') as f:
        pickle.dump(test, f)
