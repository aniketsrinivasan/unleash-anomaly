import sqlite3
import os
import pandas as pd


def get_sqlite_connection(db_path: str, verbose=True) -> sqlite3.Connection:
    """
    Connects to a SQLite3 database and returns an sqlite3.Connection object.

    :param db_path:     path to sqlite3 database.
    :param verbose:     prints debugging information.
    :return:            sqlite3.Connection object.
    """
    if verbose:
        print(f"Connection to SQLite3 database at {db_path}...")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"File {db_path} not found.")

    # Connecting to the database:
    connection = sqlite3.connect(db_path)

    if verbose:
        print(f"Connected to SQLite3 database at {db_path}", end="\n\n")
    return connection


def sqlite_to_pandas(connection: sqlite3.Connection, table_name: str,
                     timestamp=None, verbose=True) -> pd.DataFrame:
    """
    Consumes a SQLite3 connection and returns a pandas DataFrame of the provided table name.

    :param connection:  a connection to the SQLite3 database to use.
    :param table_name:  name of the table to extract as a DataFrame.
    :param timestamp:   timestamp to extract data for.
    :param verbose:     prints debugging information.
    :return:            a pd.DataFrame of the provided table name from the SQLite3 database connection.
    """
    if verbose:
        print(f"Extracting pd.DataFrame from provided SQLite3 connection for table {table_name}...")
    # Get the pd.DataFrame:
    if timestamp is None:
        # Extract all timestamps:
        dataframe = pd.read_sql_query(f"SELECT * from {table_name}", connection)
    else:
        # Get only information for the provided timestamp:
        dataframe = pd.read_sql_query(f"SELECT * from {table_name} WHERE BucketTS={timestamp}", connection)
    if verbose:
        print(f"DataFrame extracted from SQLite3 connection.")
        print(dataframe, end="\n\n")
    return dataframe


def sqlite_get_stacked_database(directory: str, table_name: str, save_path: str,
                                sort_values: bool = False, sort_column: str = None,
                                verbose=True):
    # Initializing an empty dataframes list, iterating through all files in database:
    dataframes = []
    for file in os.listdir(directory):
        this_connection = get_sqlite_connection(os.path.join(directory, file))
        this_dataframe = sqlite_to_pandas(this_connection, table_name)
        dataframes.append(this_dataframe)
        this_connection.close()
    stacked_dataframe = pd.concat(dataframes, ignore_index=True)

    if sort_values and (sort_column is not None):
        stacked_dataframe = stacked_dataframe.sort_values(by=sort_column, ascending=True)

    # Creating new SQLite3 database:
    new_connection = sqlite3.connect(save_path)
    stacked_dataframe.to_sql(table_name, new_connection, if_exists="replace")
    new_connection.close()

    return True
