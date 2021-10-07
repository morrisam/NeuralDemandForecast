from datetime import date
import datetime
from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def print_run_stats(inputfile,output_results,args):

    output_results['lstm_favor']=output_results['base_mape']-output_results['lstm_mape_score']
    output_results['lstm_win_base'] =  output_results['lstm_favor']>0
    output_results['arima_win_base'] = output_results['base_mape']-output_results['arima_mape'] > 0
    output_results['lstm_win_arima'] = output_results['arima_mape']-output_results['lstm_mape_score']  > 0

    data_name = inputfile.split('/')[-1].split('.')[0].replace(' ', '_')

    with open(f"{data_name}.txt", "a") as f:
        f.write(f"Today's date: {date.today()}\n")
        f.write(f"-------------------------\n")
        f.write(f"File input name:\n")
        f.write(f"{inputfile}:\n")
        f.write(f"-------------------------\n")
        f.write(f"args:\n")
        f.write(f"{args}:\n")
        f.write(f"-------------------------\n")
        f.write(f"summary LSTM wins BASE {output_results['lstm_win_base'].sum()} / {len(output_results)} \n")
        f.write(f"-------------------------\n")
        f.write(f"summary ARIMA wins BASE {output_results['arima_win_base'].sum()} / {len(output_results)} \n")
        f.write(f"-------------------------\n")
        f.write(f"summary LSTM wins ARIMA {output_results['lstm_win_arima'].sum()} / {len(output_results)} \n")
        f.write(f"-------------------------\n")
        wins=output_results[['arima_win_base','lstm_win_arima','lstm_win_base']].sum(axis=1)
        f.write(f"summary LSTM>ARIMA>BASE {(wins==3).sum()} / {len(output_results)} \n")
        f.write(f"-------------------------\n")
        f.write(f"lstm mape avg {output_results['lstm_mape_score'].mean():.4f} \n")
        f.write(f"baseline mape avg {output_results['base_mape'].mean():.4f} \n")
        f.write(f"arima mape avg {output_results['arima_mape'].mean():.4f} \n")
        f.write(f"-------------------------\n")
        f.write(output_results.to_string())
    output_results.to_csv(f"results_{data_name}.csv")

def result_dir(inputfile):
    data_name = inputfile.split('/')[-1].split('.')[0].replace(' ', '_')
    datestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = data_name + '_' + datestamp
    dir_path=f"results/{dir_name}"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path