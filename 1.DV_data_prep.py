"""
Author: Caroline Davidson

In this script, I prepare data on reported crimes in the City of Chicago for a 
panel regression to investigate whether domestic violence trends more closely
follow those of violent crime or property crime and whether this relationship
changed during/after the Covid-19 pandemic. I use data from 2001 - 2023. I
aggregate data by month by community area, with the option to aggregate instead
by police district.
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import itertools
import zipfile

BASE_PATH = r"/Users/Caroline/Dropbox/*Harris/Coding Samples/Chicago_DV_trends"

# PART 1: FUNCTION DEFINITION

INDEX_CATEGORY_DICT = {
    # I use the FBI's classifications for violent and property crimes, limited to index crimes.
    "violent crime": [
        # murder and nonnegligent manslaughter, rape, robbery, and aggravated assault
        "ASSAULT",
        "BATTERY",
        "CRIMINAL SEXUAL ASSAULT", # various categories of rape (with / without weapon)
        "HOMICIDE",
        "ROBBERY"
        ],
    
    "property crime": [
        # includes the offenses of burglary, larceny-theft, motor vehicle theft, and arson
        "ARSON",
        "BURGLARY",
        "MOTOR VEHICLE THEFT",
        "THEFT"
        ], 
    }


def categorize_index_codes(primary_description):
    """Classify violent and property index crimes by primary description."""
    for category, primary_descriptor_list in INDEX_CATEGORY_DICT.items():
        if primary_description in primary_descriptor_list:
            return category
    return "other"


def load_process_iucr_codes():
    """Return dictionary of IUCR codes by index crime type."""
    csv_path = os.path.join(BASE_PATH, "Data", "iucr_codes.csv")
    # iucr codes obtained from: https://dev.socrata.com/foundry/data.cityofchicago.org/c7ck-438e
    if os.path.exists(csv_path):
        print(">> Reading IUCR codes from existing csv file.")
        iucr_codes = pd.read_csv(csv_path)
    else:
        print(">> Downloading IUCR codes via API.")
        iucr_codes = pd.read_csv("https://data.cityofchicago.org/resource/c7ck-438e.csv")
        iucr_codes.to_csv(csv_path)
    
    iucr_codes["iucr"] = iucr_codes["iucr"].str.zfill(4) # add leading 0s to match yearly data format
    # categorize index codes according to FBI categorical classification
    index_codes = iucr_codes[iucr_codes["index_code"] == "I"]
    index_codes["crime_category"] = index_codes['primary_description'].apply(categorize_index_codes)

    codes_full = iucr_codes.merge(index_codes, how = "outer", indicator = True)
    codes_full["crime_category"] = codes_full["crime_category"].replace({np.nan: "non-index crime"})
    IUCR_DICT = codes_full.set_index("iucr")["crime_category"].to_dict()
    
    return IUCR_DICT


def download_classify_crime_data_by_year(year):
    """Fetch/load crime data for year and return df with index crime classification."""
    csv_name = str(year) + "_crime_data.csv"
    csv_path = os.path.join(BASE_PATH, "Data/yearly reported crime", csv_name)
    
    url = (
        "https://data.cityofchicago.org/resource/ijzp-q8t2.csv?"
        f"$where=year={year}"
        "&$limit=1000000"
        "&$select=date,iucr,domestic,community_area,district,beat,latitude,longitude"
        )
    
    if os.path.exists(csv_path):
        print(f"\n *** Reading {year} crime data from existing csv file. ***")
        df = pd.read_csv(csv_path)
    else:
        print(f"\n *** Downloading {year} crime data via API. ***")
        df = pd.read_csv(url)
        df.to_csv(csv_path)
        
    df["crime_category"] = df["iucr"].map(IUCR_DICT)
    
    # check for NAs that could cause issues in analysis
    na_count = pd.DataFrame(np.sum(df.isna(), axis = 0), columns = ["Count NAs"])
    print(f"> Count of Missing Values in raw {year} data:")
    print(na_count)
    
    return df


# Since community area is missing for some observations, I will use lat/long and
# spatial joining to identify community area when missing. The same process is
# applied for police districts as well, which are missing to a much smaller degree.


def unzip_load_shapefile(file_name, folder_name, shapefile_name):
    """Unzip shapefile and return geopandas dataframe.
    
    Arguments
    ---------
    file_name -- name of zip file containing target shapefile
    folder_name -- name of folder to create for files extracted from zip file
    shapefile_name -- name of shapefile to read after unzipping folder
    
    Returns
    -------
    geopandas dataframe of the shapefile with CRS 4326
    """
    zip_file_path = os.path.join(BASE_PATH, "Data","shapefiles", file_name)
    folder_path = os.path.join(BASE_PATH, "Data", "shapefiles", folder_name)
    
    with zipfile.ZipFile(zip_file_path, mode='r') as zip:
        zip.extractall(folder_path)
    
    df = gpd.read_file(os.path.join(folder_path, shapefile_name)).to_crs("EPSG:4326")
    return df
    

def add_comm_area_district(df): 
    """Perform spatial join to identify community areas and/or police districts if missing.
    
    Argument
    ---------
    df -- name of pandas dataframe with missing community area and/or police
    district values
    
    Returns
    -------
    geopandas dataframe (CRS 4326) with geocomputed (when missing) community area
    and police district
    """
    comm_area_na_count = np.sum(df["community_area"].isna(), axis=0)
    district_na_count = np.sum(df["district"].isna(), axis=0)            
    if (comm_area_na_count == 0) and (district_na_count == 0):
        print("> No missing community area or district values.") 
        df = df.drop(columns=["latitude", "longitude"])
        return df
    
    else: 
        geometry = [Point(lng, lat) for lng, lat in zip(df["longitude"], df["latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=4326)
        
        if comm_area_na_count > 0:
            gdf = gdf.rename(columns={"community_area": "community_area_original"})
            community_areas["area_num_1"] = community_areas["area_num_1"].astype(float)
            gdf = gdf.sjoin(community_areas[["area_num_1", 
                                             "community", 
                                             "geometry"]]).drop(columns="index_right")
            gdf["community_area"] = gdf["community_area_original"].fillna(gdf["area_num_1"]) 
            comm_area_nas_left = np.sum(gdf["community_area"].isna(), axis=0)
            print("> Number community areas still missing after spatial join:", comm_area_nas_left)
            new_df = gdf.drop(columns=["community_area_original", "latitude", "longitude",
                                        "area_num_1", "community", "geometry"]) 
            
        if district_na_count > 0:
            gdf = gdf.rename(columns={"district": "district_original"})
            gdf = gdf.sjoin(police_districts[["DISTRICT", 
                                              "NAME", 
                                              "geometry"]]).drop(columns="index_right")  
            gdf["district"] = gdf["district_original"].fillna(gdf["DISTRICT"]).astype(float)
            district_nas_left = np.sum(gdf["district"].isna(), axis=0)            
            print("> Number police districts still missing after spatial join:", district_nas_left)
            new_df = gdf.drop(columns=["district_original", "latitude", "longitude", 
                                       "DISTRICT", "NAME", "geometry"])  
    
        return new_df
 

def gen_base_df_all_combos(areas="community_area"):
    """Generate all possible combinations of geographic area, month, and crime type.
    This will be used to ensure a balanced panel with 0 crime counts where needed.
    
    Keyword argument:
    ----------------
    areas -- name of geographic variable that will be used for aggregation
    (default is community areas, other option allowed is police districts)
    
    Returns:
    -------
    pandas dataframe with all possible combinations of month, 3 crime types,
    and values of the geographic variable
    """
    if areas == "community_area":
        num_list = list(range(1, 78))
    elif areas == "district":
        # police districts numbered 1-25 except 13, 21, 23
        to_remove = [13, 21, 23]
        num_list = list(range(1, 26))
        num_list = [x for x in num_list if x not in to_remove]
    else:
        print("Choose a valid area: 'community_area' or 'district' (for police districts)")
        return
        
    combos = list(itertools.product(
        ("DV", "property crime", "violent crime"), #crime categories for regression
        num_list, # number of areas
        range(1, 13))) # months
        
    combos_df = pd.DataFrame(combos, columns=["reg_crime_group", areas, "month"])
    return combos_df


def aggregate_data_by_area_period(df, base_df, geographic_agg_level="community_area"):
    """
    Aggregate data by crime type, month, and area.
    
    Arguments:
    ---------
    df -- df with crime data to aggregate
    base_df -- pandas df whose rows correspond to desired post-aggregation rows
    (essentially establishing the multi-index result desired), generated by 
    gen_base_df_all_combos()
    geographic_agg_level -- name of variable to use for geographic aggregation
    (default is community areas, other option is police districts)
    
    Returns:
    -------
    pandas dataframe of aggregated crime data
    """
    df["datetime"] = pd.to_datetime(df["date"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
  
    df["reg_crime_group"] = np.where(df["domestic"], "DV", df["crime_category"])
    # keep only relevant categories for this analysis
    df = df[df["reg_crime_group"].isin(
        {"DV", "property crime", "violent crime"})].reset_index(drop=True)
    
    # aggregate to specified geographical level
    df_agg = df.groupby(
        ["reg_crime_group", geographic_agg_level, "month", "year"]
        ).size().reset_index(name="n_reported")
  
    if geographic_agg_level == "district":
    # drop district 31 when agg by police district (not listed by CPD as one of their 22 districts, data start 2013)
        n_district_31 = df_agg[df_agg["district"] == 31].shape[0]
        if n_district_31 > 0:
            print(f"> Dropping {n_district_31} rows corresponding to police district 31.")
            df_agg = df_agg[df_agg["district"] != 31]
    
    if geographic_agg_level == "community_area":
    # drop rows where community area = 0 (unsure why coded this way, exclude from this analysis)
        n_comm_area_0 = df_agg[df_agg["community_area"] == 0].shape[0]
        if n_comm_area_0 > 0:
            print(f"> Dropping {n_comm_area_0} rows corresponding to community area 0.")
            df_agg = df_agg[df_agg["community_area"] != 0]
    
    df_agg_full = df_agg.merge(base_df, how="right")
    df_agg_full["n_reported"] = df_agg_full["n_reported"].replace({np.nan: 0})   
    df_agg_full["year"] = df_agg_full["year"].replace({np.nan: df["year"][0]})
        
    return df_agg_full


def gen_full_df(base_df, area="community_area", start = 2001, end = 2023):
    """
    Generate dataframe of crime counts by type, area, period.
    
    Arguments:
    ---------
    base_df -- pandas df whose rows correspond to desired post-aggregation rows
    (essentially establishing the multi-index result desired), generated by 
    gen_base_df_all_combos()
    area -- name of variable to use for geographic aggregation 
    (default is community areas, other option is police districts)
    start -- first year of crime data to use (default is 2001)
    end -- last year of crime data to use (default is 2023)
    
    Returns:
    -------
    pandas dataframe of crime data aggregated by type, area, and period
    """
   
    yearly_dfs = {} # initalize dictionary to store aggregated dfs for each year

        
    for year in range(start, end+1):
        df = download_classify_crime_data_by_year(year)

        # add comm area / district # where missing
        df_geo = add_comm_area_district(df)

        if area == "community_area":
            df_agg = aggregate_data_by_area_period(df_geo, base_df)
        else:
            df_agg = aggregate_data_by_area_period(df_geo, 
                                                   base_df,
                                                   geographic_agg_level="district")
   
        yearly_dfs[f"data_{year}"] = df_agg
        
    full_df = pd.concat(yearly_dfs.values(), ignore_index=True)
    
    return full_df


# PART 2: FUNCTION EXECUTION
if __name__ == "__main__":
    IUCR_DICT = load_process_iucr_codes()

    community_areas = unzip_load_shapefile("Boundaries - Community Areas (current).zip", 
                                        "community area boundaries", 
                                        "geo_export_75cc0f3f-cd68-496a-bbb6-489e8c562a16.shp")

    police_districts = unzip_load_shapefile("Police_District_Boundary_View-shp.zip", 
                                            "police district boundaries", 
                                            "Police_District_Boundary_View.shp")

    crime_month_comm_area_combos = gen_base_df_all_combos()  
    crime_month_district_combos = gen_base_df_all_combos(areas="district")

    # create community area level dataframes
    full_df = gen_full_df(base_df=crime_month_comm_area_combos)
    full_df_annual = full_df.groupby(["reg_crime_group", 
                                    "community_area", 
                                    "year"])["n_reported"].sum().reset_index()

    # create police district area level dataframes
    full_df_district = gen_full_df(base_df=crime_month_district_combos, area="district")
    full_df_district_annual = full_df_district.groupby(["reg_crime_group",
                                                        "district",
                                                        "year"])["n_reported"].sum().reset_index()

    # save to csv for analysis in R
    full_df.to_csv(os.path.join(BASE_PATH, "Data/regression ready aggregations", "monthly_comm_area_2001_2023.csv"))
    full_df_annual.to_csv(os.path.join(BASE_PATH, "Data/regression ready aggregations", "annual_comm_area_2001_2023.csv"))
    full_df_district.to_csv(os.path.join(BASE_PATH, "Data/regression ready aggregations", "monthly_district_2001_2023.csv"))
    full_df_district_annual.to_csv(os.path.join(BASE_PATH, "Data/regression ready aggregations", "annual_district_2001_2023.csv"))
