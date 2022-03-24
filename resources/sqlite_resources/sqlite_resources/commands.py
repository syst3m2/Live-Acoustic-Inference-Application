import sqlite3

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        print(e)

    return conn


def execute_sql_command(conn, sql, entry=[]):
    cur = conn.cursor()
    cur.execute(sql, entry)
    return cur

def execute_multiple_sql_command(conn, sql, entry=[]):
    """
    Create a new task
    :param conn:
    :param sql
    :param entry:
    :return:
    """

    cur = conn.cursor()
    cur.executemany(sql, entry)
    return cur

def get_db_entries(db_file, table, start_t, stop_t):
    sql = "SELECT filename, start_time, end_time, num_samples from {} where end_time >=? and start_time <=?;".format(table) #todo: need to find a safer workaround than string formatting here, but sqlite doesn't allow ? for tables
    conn = create_connection(db_file)
    with conn:
        cur = execute_sql_command(conn, sql, (start_t, stop_t))
    return cur