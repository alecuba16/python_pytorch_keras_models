def load_wtdata(wt_query = None,db_config = None):
    import mysql.connector
    from pandas import DataFrame

    # query = "call formatter('" + table_cast_park_dic + "'," \
    #         + wt_query['wp_id'] + "," \
    #         + wt_query['ld_id'] + ",'" \
    #         + wt_query['fault'] + "','" \
    #         + wt_query['array_id_walm'] + "','" \
    #         + wt_query['array_ot'] + "','" \
    #         + wt_query['power_condition'] + "','" \
    #         + wt_query['include_variables'] + "','" \
    #         + wt_query['exclude_variables'] + "'," \
    #         + wt_query['unix_timestamp_ini'] + "," \
    #         + wt_query['unix_timestamp_end'] + "," \
    #         + str(int(wt_query['freq_dat_med_min']) * 60) + ",0)"
    #
    query = [db_config['table_cast_park_dic'],wt_query['wp_id'],wt_query['ld_id'],wt_query['fault'],wt_query['array_id_walm'],wt_query['array_ot'],wt_query['power_condition'],wt_query['include_variables'],wt_query['exclude_variables'],wt_query['unix_timestamp_ini'],wt_query['unix_timestamp_end'],wt_query['freq_dat_med_min'] * 60,0]

    # Connect
    cnx = mysql.connector.connect(user=db_config['user'], password=db_config['password'],
                                  host=db_config['host'],
                                  database=db_config['db'])
    try:
       cursor = cnx.cursor()
       cursor.callproc('formatter',query)
       print("query done.. processing data...")
       for result in cursor.stored_results():
           results = result.fetchall()
       df = DataFrame(results)
       df.columns = result.column_names
    finally:
        cnx.close()
    return df
