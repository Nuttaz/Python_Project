# Resale Flat Market Analysis in Singapore

## Overview

Welcome to my analysis of the resale flat market in Singapore, specifically focused on investment opportunities. This project was developed with the goal of better understanding the resale flat market and identifying the best investment options. It examines the high demand and market growth to provide insights into optimal investment choices.

The data used for this analysis is sourced from [data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view?dataExplorerPage=1), which serves as the foundation of my work. This dataset includes detailed information on town, flat type, room size, lease commencement year, resale price, storey range, and more. By utilizing a series of Python scripts, I explore important questions such as the most in-demand towns, flat types, and the overall growth trends in the market.

## The Questions

Here are the key questions I aim to answer through this project:

1. **What is the relationship between resale price and remaining lease duration?**
2. **What are the best options for resale flat investments in Singapore?**

## Tools I Used

To dive deep into the resale flat market, I used several powerful tools for data analysis:

- **Python**: The main tool for data analysis, enabling me to process the dataset and uncover important insights. Additionally, I used the following Python libraries:
    - **Pandas Library**: Used for data manipulation and analysis.
    - **Matplotlib Library**: Employed for visualizing the data.
    - **Seaborn Library**: Assisted in creating advanced visualizations.
    - **Datasets Library**: Used to import data into a DataFrame.
- **Jupyter Notebooks**: This tool allowed me to run Python scripts efficiently, with the added benefit of incorporating notes and analyses.
- **Visual Studio Code**: My preferred environment for executing Python scripts.
- **Anaconda3**: Used to create a suitable environment for the project.
- **Git & GitHub**: These tools were crucial for version control and sharing the Python code and analysis.

## Data Preparation and Cleanup

This section describes the steps I took to prepare and clean the data for analysis, ensuring both accuracy and usability.

### Importing & Cleaning the Data

The process starts with importing the necessary libraries and loading the dataset. I then examine the first five rows to better understand the structure of the data. I ensure that each column is assigned the correct data type and remove any duplicate records to clean the data.

Additionally, I add more fields to the existing DataFrame to support the analysis further.

```python
# Importing Libraries
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
import seaborn as sns
```
```python
# Import dataframe Resale flat prices based on registration date from Jan-2017 onward (Mar-2025)
df = pd.read_csv(r'C:\Users\1\Desktop\Nut\Data_Analyst\Python_Project\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv', skiprows=0)
```
```python
# Display the first 5 rows of the dataframe
df.head()
```
```python
# Data cleaning and prepare calculation
df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
df['year'] = df['month'].dt.year
df['price_sqm'] = df['resale_price'] / df['floor_area_sqm'] 

# Function to convert 'years months' to total months
def convert_to_months(lease):
    # Initialize years and months as 0
    years = 0
    months = 0
    
    # If 'years' is found in the string
    if 'year' in lease:
        years = int(lease.split('year')[0].strip())  # Extract years
        
    # If 'months' is found in the string
    if 'month' in lease:
        months = int(lease.split('month')[0].split()[-1].strip())  # Extract months
    
    # Calculate total months (years * 12 + months)
    return (years * 12) + months

# Apply the conversion function to 'remaining_lease'
df['remaining_lease_months'] = df['remaining_lease'].apply(convert_to_months)
df['price_sqm'] = df['price_sqm'] / df['remaining_lease_months'] 
df_unique = df.drop_duplicates().dropna(subset=['resale_price'])

print('Length of original df       :', len(df))
print('Length of drop duplicates df:', len(df_unique))
print('Rows Dropped                :', len(df)-len(df_unique))
```

## The Analysis
Each Jupyter notebook for this project aimed at investigating specific aspects of the resale flat market. Here's how I approached each question:

### 1. What is the relationship between resale price and remaining lease duration?

To find this relationship, I plot the scattered chart to see trends

```python
#Relationship between Price per sqm and Remaining Lease Months in SG

fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=150)
axes = axes.flatten()

df_overview = df_unique.copy()

# Remaining Lease Months vs Lease Commence Date
ax = axes[0]
ax.scatter(df_overview['remaining_lease_months'], df_overview['lease_commence_date'], color='skyblue', s=15)
ax.set_title('Remaining Lease Months vs Lease Commence Date')
ax.set_xlabel('Remaining Lease Months')
ax.set_ylabel('Lease Commence Date')

#----------------------------------------------------------------------------------------------------------------------

# Price per sqm vs Lease Commence Date
ax = axes[1]
ax.scatter(df_overview['price_sqm'], df_overview['lease_commence_date'], color='skyblue', s=10)
ax.set_title('Price per sqm vs Lease Commence Date')
ax.set_xlabel('Price per sqm')
ax.set_ylabel('Lease Commence Date')
# Check the y-axis range for lease commence date
y_min, y_max = ax.get_ylim()  # Get the current y-axis limits

# Define the range for lease commence date (between 2017 and 2021)
y1_start = 2016.5
y1_end = 2021.5

# Ensure that the years 2015 to 2020 are within the y-range
if y_min <= y1_start <= y_max and y_min <= y1_end <= y_max:
    # Get the x-axis range for the plot
    x_min, x_max = ax.get_xlim()  # Get the current x-axis limits
    
    # Add a red box from 2015 to 2020 in the y-axis and covering the entire x-axis
    ax.add_patch(Rectangle((x_min, y1_start), x_max - x_min, y1_end - y1_start,
                           linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
    ax.text(15, (y1_start + y1_end) / 2,
            '2017-2021', ha='left', va='center', fontsize=12, color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
    
# Define the range for lease commence date (between 2011 and 2016)
y2_start = 2011.5
y2_end = 2016.5

# Ensure that the years 2015 to 2020 are within the y-range
if y_min <= y2_start <= y_max and y_min <= y2_end <= y_max:
    # Get the x-axis range for the plot
    x_min, x_max = ax.get_xlim()  # Get the current x-axis limits
    
    # Add a red box from 2015 to 2020 in the y-axis and covering the entire x-axis
    ax.add_patch(Rectangle((x_min, y2_start), x_max - x_min, y2_end - y2_start,
                           linewidth=2, edgecolor='Green', facecolor='none', linestyle='--'))
    ax.text(15, (y2_start + y2_end) / 2,
            '2012-2016', ha='left', va='center', fontsize=12, color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

#----------------------------------------------------------------------------------------------------------------------

plt.tight_layout()

fig.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9, hspace=0.6, wspace=0.3)

fig.suptitle("Relationship between Price per sqm and Remaining Lease Months in SG", fontsize=18, fontweight='bold', y=0.92)

fig.text(0.5, 0.05, 
         'All resale flats have a leasehold term of 99 years, but they are typically resald after 4 years of living.\n As a result, the price per square meter relative to the remaining lease decreases between the 5th and 8th years,\n then stabilizes again after the 9th year.', 
         ha='center', va='center', fontsize=12, color='black',
         bbox=dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.3'))

plt.show()
```
![](image\1_price_lease.png)


### 2. What are the best options for resale flat investments in Singapore?

Based on the results from Question 1, I’ve narrowed the resale registration dates to 2020-2024, as 2025 only has three months of data. I wanted to ensure the data is up-to-date within the past five years.
Additionally, I’ve limited the lease commencement year range to 2012-2016, as the price per square meter during this period has remained stable, making it a strong option for investment.

To identify the best investment opportunities, I filtered the data to highlight the top five towns in terms of value, volume, and growth, while also considering factors like price range and room size. This query highlights the most popular towns, guiding investors on where to focus their attention

```python
#Best Town to invest in SG
fig, axes = plt.subplots(3, 2, figsize=(12, 8), dpi=150)
axes = axes.flatten()

# Customizing the x-axis formatting
def custom_xaxis_format(x, pos):
    if x >= 1000000:
        return f'{int(x / 100000000):,}B'  # For values >= 1,000,000, show 'B'
    if x >= 1000000:
        return f'{x / 1000000:,}M'  # For values >= 1,000,000, show 'M'
    if x >= 2000:
        return f'{int(x / 1000):,}K'  # For values >= 1000, show 'K'
    else:
        return f'{int(x):,}'  # For values < 1000, show normally

#------------------------------------------------------------------------------------------------------------------------

# Top 5 town Resales Values (2020-2024)
df_grouped_town = df_town.groupby('town')['resale_price'].sum().reset_index() 
top_5_town_value = df_grouped_town.sort_values(by='resale_price', ascending=False).head(5)['town'].tolist()
total_resale_price = df_grouped_town['resale_price'].sum() 
df_grouped_town['percentage'] = (df_grouped_town['resale_price'] / total_resale_price) * 100 
df_top_5_town = df_grouped_town[df_grouped_town['town'].isin(top_5_town_value)].sort_values(by='resale_price', ascending=False)

highlight_row = df_grouped_town[df_grouped_town['town'] == 'SENGKANG'].iloc[0]
highlight_value = highlight_row['resale_price']
highlight_percentage = highlight_row['percentage']

colors = ['lightcoral' if town == 'SENGKANG'
          else 'lightgreen' if town == 'YISHUN'
          else 'skyblue'
          for town in df_top_5_town['town']]

ax = axes[0]

ax.barh(df_top_5_town['town'], df_top_5_town['resale_price'], color=colors)
ax.set_title('Top 5 Town Resale Values (2020-2024)')
ax.set_xlabel('Resale Flats Value', fontsize=8)
ax.set_ylabel('')
ax.invert_yaxis()  # Invert y-axis for proper order
ax.xaxis.set_major_formatter(FuncFormatter(custom_xaxis_format))
ax.tick_params(axis='y', labelsize=8)
ax.annotate(f"Rank 2 Resales Values ({highlight_percentage:.1f}%)", 
            xy=(0, 0),
            xytext=(highlight_value, 1),
            fontsize=10, color='black', ha='right', va='center')

#-------------------------------------------------------------------------------------------------------------

# Top 5 town Resales Volume (2020-2024)
df_grouped_town = df_town.groupby('town').size().reset_index(name='count') 
top_5_town_volume = df_grouped_town.sort_values(by='count', ascending=False).head(5)['town'].tolist()
total_count = df_grouped_town['count'].sum() 
df_grouped_town['percentage'] = (df_grouped_town['count'] / total_count) * 100 
df_top_5_town = df_grouped_town[df_grouped_town['town'].isin(top_5_town_volume)].sort_values(by='count', ascending=False)

highlight_row = df_grouped_town[df_grouped_town['town'] == 'SENGKANG'].iloc[0]
highlight_value = highlight_row['count']
highlight_percentage = highlight_row['percentage']

colors = ['lightcoral' if town == 'SENGKANG'
          else 'lightgreen' if town == 'YISHUN'
          else 'skyblue'
          for town in df_top_5_town['town']]

ax = axes[1]
ax.barh(df_top_5_town ['town'], df_top_5_town ['count'], color=colors)
ax.set_title('Top 5 town Resales Volume (2020-2024)')
ax.set_xlabel('Count of Resale Flats',fontsize=8)
ax.set_ylabel('')
ax.invert_yaxis()
ax.xaxis.set_major_formatter(FuncFormatter(custom_xaxis_format))
ax.tick_params(axis='y', labelsize=8)
ax.annotate(f"Rank 2 Resales Volume ({highlight_percentage:.1f}%)", 
            xy=(0, 0),
            xytext=(highlight_value, 1),
            fontsize=10, color='black', ha='right', va='center')

#----------------------------------------------------------------------------------------------------------------------

# 2020 to 2024 Top 5 town Resales price per Sqm per lease remain Growth (%)
y2020_y2024 = [2020, 2024]
df_filtered = df_town[df_town['year'].isin(y2020_y2024)]
pivot_y20_vs_y24 = df_filtered.pivot_table(index='year', columns='town', values='price_sqm', aggfunc='median')
growth_y20_vs_y24 = pivot_y20_vs_y24.pct_change(axis=0, fill_method=None) * 100
growth_y20_vs_y24 = growth_y20_vs_y24.drop(index=2020, axis=0).transpose().sort_values(2024, ascending=False).reset_index().head(5)
df_top_5_town = growth_y20_vs_y24.copy()
top_5_town_growth_y20_vs_y24 = growth_y20_vs_y24['town'].tolist()

highlight_row = growth_y20_vs_y24[growth_y20_vs_y24['town'] == 'SENGKANG'].iloc[0]
highlight_value = highlight_row[2024]

colors = ['lightcoral' if town == 'SENGKANG'
          else 'lightgreen' if town == 'YISHUN'
          else 'skyblue'
          for town in df_top_5_town['town']]

ax = axes[2]
ax.barh(df_top_5_town['town'], df_top_5_town[2024], color=colors)
ax.set_title('2020 to 2024 Top 5 town Resales price per Sqm per lease remain Growth (%)',fontsize=9)
ax.set_xlabel('Growth (%)',fontsize=8)
ax.set_ylabel('')
ax.invert_yaxis()
ax.xaxis.set_major_formatter(FuncFormatter(custom_xaxis_format))
ax.tick_params(axis='y', labelsize=8)
ax.annotate(f"Rank 3 growth ({highlight_value:.1f}%)", 
            xy=(0, 0),
            xytext=(highlight_value, 2),
            fontsize=10, color='black', ha='right', va='center')

#----------------------------------------------------------------------------------------------------------------------

# Plot the 4 Room price per sqm percentage growth from 2020 to 2024 (5 years)
highlight_town = df_town[df_town['town'].isin(top_5_town_volume) & df_town['town'].isin(top_5_town_value) & df_town['town'].isin(top_5_town_growth_y20_vs_y24)]['town'].unique().tolist()
df_filtered = df_town.copy()
df_filtered['town'] = df_filtered['town'].apply(lambda x: x if x in highlight_town else 'Average')

pivot_growth_df = df_filtered.pivot_table(index='year', columns='town', values='price_sqm', aggfunc='median')
growth_yearly_df = pivot_growth_df.pct_change(axis=0, fill_method=None) * 100
growth_yearly_df = growth_yearly_df.dropna()
growth_yearly_df = growth_yearly_df[sorted(growth_yearly_df.columns, reverse=True)]

ax = axes[3]

color_map = {'SENGKANG': 'lightcoral','YISHUN': 'lightgreen'}

for town in growth_yearly_df.columns:
    colors = color_map.get(town,'skyblue')
    ax.plot(growth_yearly_df.index, growth_yearly_df[town], label=town, color=colors)
ax.set_title('Yearly Price per Sqm per lease remain Growth %', fontsize=10)
ax.set_xlabel('')
ax.set_ylabel('Growth (%)',fontsize=8)
ax.set_ylim(0,15)
ax.set_xticks(growth_yearly_df.index)
ax.legend(loc='lower left', fontsize=8)
ax.text(0.7, 0.8,
        'SENGKANG\'s growth better than average',
        fontsize=7, color='black', ha='center', va='bottom',
        transform=ax.transAxes)

#----------------------------------------------------------------------------------------------------------------------

# Resales flat Price per Sqm per lease remain Range (2020-2024)

df_filtered = df_town.copy()
df_filtered['town'] = df_filtered['town'].apply(lambda x: x if x in highlight_town else 'Average')

ax = axes[4]

colors = ['skyblue','lightcoral','lightgreen']
sns.boxplot(x='price_sqm', y='town', data=df_filtered, ax=ax, orient='h', palette=colors, hue='town')
ax.set_title('Resales flat Price per Sqm per lease remain Range (2020-2024)',fontsize=10)
ax.set_xlabel('price per sqm per lease remain',fontsize=8)
ax.set_ylabel('')
ax.invert_yaxis()
ax.xaxis.set_major_formatter(FuncFormatter(custom_xaxis_format))

#----------------------------------------------------------------------------------------------------------------------

# Resales flat Sqm Range (2020-2024)

df_filtered = df_town.copy()
df_filtered['town'] = df_filtered['town'].apply(lambda x: x if x in highlight_town else 'Average')

ax = axes[5]

colors = ['skyblue','lightcoral','lightgreen']
sns.boxplot(x='floor_area_sqm', y='town', data=df_filtered, ax=ax, orient='h', palette=colors, hue='town')
ax.set_title('Resales flat Sqm Range (2020-2024)')
ax.set_xlabel('sqm',fontsize=8)
ax.set_ylabel('')
ax.invert_yaxis()
ax.set_xlim(0, )
ax.xaxis.set_major_formatter(FuncFormatter(custom_xaxis_format))

#----------------------------------------------------------------------------------------------------------------------

plt.tight_layout()

fig.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9, hspace=0.6, wspace=0.3)

fig.suptitle("Best Town to invest in SG", fontsize=24, fontweight='bold', y=0.92)

plt.show()
```
![](image\2_town.png)

Sengkang is the second most popular town in Singapore for resale flats. It has experienced significant growth in resale prices, with a 47% increase and an average annual growth of around 10%. The price range is slightly lower than the average, while the room sizes are larger than typical flats.

Here is addition information from ChatGPT

Sengkang Town in Singapore is an attractive option for investing in resale flats due to its excellent connectivity, including MRT and LRT links, and proximity to key developments like the North-South Corridor. With a growing population, family-friendly amenities, and ongoing infrastructure improvements, Sengkang is becoming increasingly popular for both residents and investors. The area offers more affordable prices compared to central locations, with strong potential for capital appreciation as it matures. Additionally, the town has a vibrant community, a strong rental market, and long-term development prospects, making it a solid choice for property investment.

After I got the result, I filtered down and do the same with flat_type, room size and storey.
You can check the full code [HERE](https://github.com/Nuttaz/Python_Project/blob/main/Resale%20flat%20prices%20based%20on%20registration%20date%20from%20Jan-2017%20onwards.ipynb)

![](image\3_flat_type.png)

![](image\4_room_size.png)

![](image\5_storey.png)

```python
# Best Price Range for a 92 sqm, 4-room unit on the 10th to 12th floors to invest in Sengkang
fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)
axes = axes.flatten()

# Customizing the x-axis formatting
def custom_xaxis_format(x, pos):
    if x >= 1000000:
        return f'{x / 100000000:,}B'
    if x >= 1000000:
        return f'{x / 1000000:,}M'
    if x >= 2000:
        return f'{int(x / 1000):,}K'
    else:
        return f'{int(x):,}'

ax = axes[0]
ax.scatter(df_final['price_sqm'], df_final['lease_commence_date'])
ax.set_title('Price per sqm per lease remain vs Lease Commence Date',fontsize=9)
ax.set_xlabel('Price per sqm per lease remain', fontsize = 8)
ax.set_ylabel('Lease Commence Date', fontsize = 8)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

#----------------------------------------------------------------------------------------------------------------------

df_filtered = df_final[(df_final['lease_commence_date'] == 2015) & (df_final['year'] == 2024)] # to narrow down to best lease commence date and the most current resales price
ax = axes[1]
ax.scatter(df_filtered['remaining_lease_months'], df_filtered['resale_price'] )
ax.set_title('Lease remain vs Resales price (Commence date 2015) on 2024',fontsize=9)
ax.set_xlabel('lease remain (month)', fontsize = 8)
ax.set_ylabel('resales price', fontsize = 8)
ax.yaxis.set_major_formatter(FuncFormatter(custom_xaxis_format))
ax.xaxis.set_major_formatter(FuncFormatter(custom_xaxis_format))

y_start = 600000
y_end = 650000
x_start = 1085
x_end = df_filtered['remaining_lease_months'].max()

ax.add_patch(Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                           linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
ax.text((x_start + x_end) / 2, y_end+30000,
            (f'$\\mathbf{{Range\\ to\\ Invest}}$\n price {int(y_start/1000):.0f}K - {int(y_end/1000):.0f}K\n lease remain\n {x_start:,.0f} - {x_end:,.0f} months'), ha='center', va='center', fontsize=6, color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

#----------------------------------------------------------------------------------------------------------------------

plt.tight_layout()

fig.subplots_adjust(top=0.83, bottom=0.4, left=0.1, right=0.9, wspace=0.3)

fig.suptitle("Best Price Range for a 92 sqm, 4-room unit on the 10th to 12th floors to invest in Sengkang", fontsize=12, fontweight='bold', y=0.92)

plt.show()
```
![](image\6_final.png)

## Conclusion of the analysis 

The best choice for investment in resale flats in Singapore is a 92 sqm, 4-room unit on the 10th to 12th floors in Sengkang, with a lease that commenced in 2015. The price you should aim for is around 600K-650K, with the maximum lease remaining.

## What I Learned
Throughout this project, I gained a deeper understanding of the resale flat market in Singapore, where leasehold properties are influenced by the remaining lease period, unlike in my hometown, where properties are freehold. I also enhanced my technical skills in Python especually in data manipulation and visualization. Here are a few specific things I learned:

- **Advanced Python Usage**: Throughout this project, I gained hands-on experience using advanced Python libraries such as Pandas for efficient data manipulation, Seaborn and Matplotlib for creating insightful data visualizations, and several other specialized libraries. These tools enabled me to perform complex data analysis tasks with much greater efficiency, allowing me to uncover patterns and insights that would have been time-consuming and challenging to identify manually.

- **The Importance of Data Cleaning**: One of the most valuable lessons I learned was the critical role of thorough data cleaning and preparation. Before any meaningful analysis can be conducted, ensuring the data is accurate, complete, and properly formatted is essential. This process not only helps in minimizing errors but also plays a significant role in ensuring the reliability of the insights derived from the data, making the entire analysis more trustworthy.

- **Complex Data Visualization**: I gained valuable experience in creating advanced visualizations, learning how to combine multiple charts into a single cohesive figure. Additionally, I enhanced the clarity of the charts by using annotations, text labels, and color to highlight specific key insights. I customized axis scales based on the data values, ensuring that the visuals accurately represented the information. By fine-tuning the entire figure layout, I was able to balance the titles, legends and individual charts, resulting in a more organized and informative presentation of the data

- **Strategic Skill Analysis**: I also realized the importance of understanding the dataset and framing the right questions before diving into analysis. By taking the time to carefully analyze the structure of the data and asking the right questions upfront, I was able to set a clear and strategic direction for my analysis. This step significantly improved the efficiency of the process, enabling me to focus on the most relevant aspects and uncover meaningful insights

## Conclusion

This analysis of the resale flat market in Singapore has provided valuable insights into the current trends and investment opportunities. By examining key factors such as resale price, lease duration, and location, we’ve identified optimal investment choices that offer the highest potential returns. The analysis revealed that a 92 sqm, 4-room unit on the 10th to 12th floors in Sengkang, with a lease commencement in 2015, is one of the best options for investment, particularly within the price range of 600K-650K.

Throughout this project, I not only deepened my understanding of the dynamics of the Singapore resale flat market but also strengthened my data analysis and visualization skills. By applying advanced Python techniques, I was able to manipulate and analyze the data efficiently, uncovering patterns and insights that may have been difficult to spot otherwise. Additionally, the importance of data cleaning, careful question formulation, and effective visualization became clear as crucial steps in the analytical process.

In conclusion, this project has reinforced the value of a structured and methodical approach to data analysis, while also highlighting the potential of the resale flat market in Singapore for investors who make informed, strategic decisions. Moving forward, these insights will be essential for anyone looking to navigate and invest in the evolving real estate market of Singapore.