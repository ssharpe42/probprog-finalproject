import pandas as pd
import sqlite3
import pickle


def get_hourly_trips(db_file):

    conn = sqlite3.connect(db_file)

    # Get all trips
    trips = pd.read_sql_query("select * from trip;", con=conn)

    # Convert to datetime
    trips.loc[:, trips.columns.str.contains('date')] = \
        trips.loc[:, trips.columns.str.contains('date')].apply(pd.to_datetime)

    # Convert to hour
    trips['hour'] = trips['start_date'].dt.hour
    trips['weekday'] = trips['start_date'].dt.weekday
    trips['month'] = trips['start_date'].dt.month
    trips['date'] = trips['start_date'].dt.date

    # Trips by station and hour
    hourly_trips = (trips.groupby(['start_station_id',
                                   'start_station_name',
                                   'date',
                                   'month',
                                   'weekday',
                                   'hour'])
                    .size()
                    .reset_index(name='demand'))

    return hourly_trips


if __name__ == '__main__':

    DB_FILE = 'data/sf_bikeshare.sqlite'
    hourly_trips = get_hourly_trips(DB_FILE)

    with open('data/hourly_trips.pickle', 'wb') as f:
        pickle.dump(hourly_trips, f)
