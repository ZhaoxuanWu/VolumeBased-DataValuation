Context
This dataset expands on my earlier New York City Census Data dataset. It includes data from the entire country instead of just New York City. The expanded data will allow for much more interesting analyses and will also be much more useful at supporting other data sets.

Content
The data here are taken from the DP03 and DP05 tables of the 2015 American Community Survey 5-year estimates. The full datasets and much more can be found at the American Factfinder website. Currently, I include two data files:

acs2015censustract_data.csv: Data for each census tract in the US, including DC and Puerto Rico.
acs2015countydata.csv: Data for each county or county equivalent in the US, including DC and Puerto Rico.
The two files have the same structure, with just a small difference in the name of the id column. Counties are political subdivisions, and the boundaries of some have been set for centuries. Census tracts, however, are defined by the census bureau and will have a much more consistent size. A typical census tract has around 5000 or so residents.

The Census Bureau updates the estimates approximately every year. At least some of the 2016 data is already available, so I will likely update this in the near future.

Acknowledgements
The data here were collected by the US Census Bureau. As a product of the US federal government, this is not subject to copyright within the US.

Inspiration
There are many questions that we could try to answer with the data here. Can we predict things such as the state (classification) or household income (regression)? What kinds of clusters can we find in the data? What other datasets can be improved by the addition of census data?


## URL
https://www.kaggle.com/muonneutrino/us-census-demographic-data


## Files to download

- acs2015_census_tract_data.csv
- acs2015_county_data.csv
- acs2017_census_tract_data.csv
- acs2017_county_data.csv