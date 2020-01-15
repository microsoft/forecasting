"""
This script downloads the SMD Hourly Data from 2011 to 2017
from https://www.iso-ne.com/isoexpress/web/reports
/load-and-demand/-/tree/zone-info
"""


import os
from urllib.request import urlretrieve
import pandas as pd
from benchmark_paths import DATA_DIR

DATA_FILE_LIST_NEW_FORMAT = ["2016_smd_hourly.xls", "2017_smd_hourly.xlsx"]
SHEET_LIST = ["ME", "NH", "VT", "CT", "RI", "SEMASS", "WCMASS", "NEMASSBOST"]
SHEET_LIST_NEW = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA"]
COLUMN_LIST = ["Date", "Hour", "DEMAND", "DryBulb", "DewPnt"]
COLUMN_LIST_NEW = ["Date", "Hr_End", "RT_Demand", "Dry_Bulb", "Dew_Point"]

NUM_TRY = 10


def validate_file(fpath, fname):
    """
    This helper function validates that specified xls file is valid, that is, that it contains expected data.
    """
    xls = pd.ExcelFile(fpath)

    if fname in DATA_FILE_LIST_NEW_FORMAT:
        sheet_list_cur = SHEET_LIST_NEW
    else:
        sheet_list_cur = SHEET_LIST

    for sheet_name in sheet_list_cur:
        df = pd.read_excel(xls, sheet_name)
        if fname in DATA_FILE_LIST_NEW_FORMAT:
            df = df[COLUMN_LIST_NEW]
        else:
            df = df[COLUMN_LIST]

        for c in df.columns:
            if all(df[c].isnull()):
                return False

    return True


def download_data():
    """Main function that downloads the SMD data from the urls in urls variable.
    
    Raises:
        Exception: if file at the specified url is not valid.
    """

    urls = [
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2011_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2012_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2013_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2015/05/2014_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2015/02/smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx",
    ]

    for url in urls:

        url_tokens = url.split("/")
        fname = url_tokens[-1]

        # add prefix to 2015 and 2016 file names
        year_month = url_tokens[-3:-1]
        year_month = "/".join(year_month)
        print(year_month)
        if year_month in ["2015/02", "2016/02"]:
            fname = url_tokens[-3] + "_" + fname

        fpath = os.path.join(DATA_DIR, fname)

        # Check if file already exists and skip if so
        if os.path.exists(fpath):
            print(fpath + " already exists")
            continue

        print("Downloading", url)

        file_valid = False
        for i in range(NUM_TRY):
            f, _ = urlretrieve(url, fpath)
            print("Downloaded to", fpath)

            if validate_file(fpath, fname):
                file_valid = True
                break
            else:
                print("Downloaded file is not valid, retrying... {} retries " "left".format(NUM_TRY - i - 1))

        if not file_valid:
            raise Exception("Unable to download valid load data from {}. " "Please try again later.".format(url))


if __name__ == "__main__":
    download_data()
