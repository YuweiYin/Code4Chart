import os
from typing import Optional, Dict, List


class DefaultInputs:

    def __init__(
            self,
            project_root_dir: Optional[str] = None,
    ):
        # Data and checkpoint directory
        self.project_root_dir = project_root_dir
        self.home_dir = os.path.expanduser("~")
        self.data_dir = os.path.join(project_root_dir, "data/code4chart")
        # self.ckpt_dir = os.path.join(project_root_dir, "ckpt/code4chart")
        os.makedirs(self.data_dir, exist_ok=True)
        # os.makedirs(self.ckpt_dir, exist_ok=True)

        self.data_dir_raw = os.path.join(self.data_dir, "raw")
        # self.data_dir_process = os.path.join(self.data_dir, "process")
        os.makedirs(self.data_dir_raw, exist_ok=True)
        # os.makedirs(self.data_dir_process, exist_ok=True)

        # Tabular datasets:
        # https://www.kaggle.com/datasets?sort=votes
        # // https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        self.datasets_info = [
            {
                "id": "1",
                "url": "https://www.kaggle.com/datasets/imdevskp/corona-virus-report",
                "filepath": os.path.join(self.data_dir_raw, "COVID-19_Dataset.csv"),  # covid_19_clean_complete.csv
                "filename": "COVID-19_Dataset.csv",
                "name": "COVID-19 Dataset",
                "description": """
Context: A new coronavirus designated 2019-nCoV was first identified in Wuhan, the capital of China's Hubei province. \
People developed pneumonia without a clear cause and for which existing vaccines or treatments were not effective. \
The virus has shown evidence of human-to-human transmission. \
Transmission rate (rate of infection) appeared to escalate in mid-January 2020. \
As of 30 January 2020, approximately 8,243 cases have been confirmed.
Content: Day to day country wise number of cases.
                """.strip(),
            },
            {
                "id": "2",
                "url": "https://www.kaggle.com/datasets/gregorut/videogamesales",
                "filepath": os.path.join(self.data_dir_raw, "Video_Game_Sales.csv"),
                "filename": "Video_Game_Sales.csv",
                "name": "Video Game Sales",
                "description": """
Context: This dataset contains a list of video games with sales greater than 100,000 copies. \
It was generated by a scrape of vgchartz.com.
Content: Fields include: \
Rank - Ranking of overall sales; \
Name - The games name; \
Platform - Platform of the games release (i.e. PC,PS4, etc.); \
Year - Year of the game's release; \
Genre - Genre of the game; \
Publisher - Publisher of the game; \
NA_Sales - Sales in North America (in millions); \
EU_Sales - Sales in Europe (in millions); \
JP_Sales - Sales in Japan (in millions); \
Other_Sales - Sales in the rest of the world (in millions); \
Global_Sales - Total worldwide sales.
                """.strip(),
            },
            {
                "id": "3",
                "url": "https://www.kaggle.com/datasets/uciml/iris",
                "filepath": os.path.join(self.data_dir_raw, "Iris_Species.csv"),
                "filename": "Iris_Species.csv",
                "name": "Iris Species",
                "description": """
Context: The Iris dataset was used in R.A. Fisher's classic 1936 paper, \
The Use of Multiple Measurements in Taxonomic Problems, \
and can also be found on the UCI Machine Learning Repository.
Content: It includes three iris species with 50 samples each as well as some properties about each flower. \
One flower species is linearly separable from the other two, \
but the other two are not linearly separable from each other.
                """.strip(),
            },
            {
                "id": "4",
                "url": "https://www.kaggle.com/datasets/shivamb/netflix-shows",
                "filepath": os.path.join(self.data_dir_raw, "Netflix_Movies_and_TV_Shows.csv"),
                "filename": "Netflix_Movies_and_TV_Shows.csv",
                "name": "Netflix Movies and TV Shows",
                "description": """
Context: Netflix is one of the most popular media and video streaming platforms. \
They have over 8000 movies or tv shows available on their platform, as of mid-2021, \
they have over 200M Subscribers globally.
Content: This tabular dataset consists of listings of all the movies \
and tv shows available on Netflix, along with details such as - \
cast, directors, ratings, release year, duration, etc.
                """.strip(),
            },
            {
                "id": "5",
                "url": "https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data",
                "filepath": os.path.join(self.data_dir_raw, "New_York_City_Airbnb_Open_Data_2019.csv"),
                "filename": "New_York_City_Airbnb_Open_Data_2019.csv",
                "name": "New York City Airbnb Open Data 2019",
                "description": """
Context: Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities \
and present more unique, personalized way of experiencing the world. \
This dataset describes the listing activity and metrics in NYC, NY for 2019.
Content: This data file includes all needed information to find out more about hosts, \
geographical availability, necessary metrics to make predictions and draw conclusions.
                """.strip(),
            },
            {
                "id": "6",
                "url": "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset",
                "filepath": os.path.join(self.data_dir_raw, "Stroke_Prediction_Dataset.csv"),
                "filename": "Stroke_Prediction_Dataset.csv",
                "name": "Stroke Prediction Dataset",
                "description": """
Context: According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, \
responsible for approximately 11% of total deaths.
Content: This dataset is used to predict whether a patient is likely to get stroke based on the input parameters \
like gender, age, various diseases, and smoking status. \
Each row in the data provides relevant information about the patient.
                """.strip(),
            },
            {
                "id": "7",
                "url": "https://www.kaggle.com/datasets/spscientist/students-performance-in-exams",
                "filepath": os.path.join(self.data_dir_raw, "Students_Performance_in_Exams.csv"),
                "filename": "Students_Performance_in_Exams.csv",
                "name": "Students Performance in Exams",
                "description": """
Context: Marks secured by the students.
Content: This data set consists of the marks secured by the students in various subjects.
                """.strip(),
            },
            {
                "id": "8",
                "url": "https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016",
                "filepath": os.path.join(self.data_dir_raw, "Suicide_Rates_Overview_1985_to_2016.csv"),
                "filename": "Suicide_Rates_Overview_1985_to_2016.csv",
                "name": "Suicide Rates Overview (1985 to 2016)",
                "description": """
Content: This compiled dataset pulled from four other datasets linked by time and place, \
and was built to find signals correlated to increased suicide rates among different cohorts globally, \
across the socio-economic spectrum.
                """.strip(),
            },
            {
                "id": "9",
                "url": "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata",
                "filepath": os.path.join(self.data_dir_raw, "TMDB_5000_Movie_Dataset.csv"),
                "filename": "TMDB_5000_Movie_Dataset.csv",
                "name": "TMDB 5000 Movie Dataset",
                "description": """
Context: What can we say about the success of a movie before it is released? \
Are there certain companies (Pixar?) that have found a consistent formula? \
Given that major films costing over $100 million to produce can still flop, \
this question is more important than ever to the industry. Film aficionados might have different interests. \
Can we predict which films will be highly rated, whether or not they are a commercial success? \
This is a great place to start digging in to those questions, \
with data on the plot, cast, crew, budget, and revenues of several thousand films.
Content: This dataset has removed the original version of this dataset per a DMCA takedown request from IMDB. \
In order to minimize the impact, this dataset is replacing it with a similar set of films and \
data fields from The Movie Database (TMDb) in accordance with their terms of use. \
The bad news is that kernels built on the old dataset will most likely no longer work.
                """.strip(),
            },
            {
                "id": "10",
                "url": "https://www.kaggle.com/datasets/unsdsn/world-happiness",
                # os.path.join(self.data_dir_raw, "World_Happiness_Report_2015.csv"),
                # os.path.join(self.data_dir_raw, "World_Happiness_Report_2016.csv"),
                # os.path.join(self.data_dir_raw, "World_Happiness_Report_2017.csv"),
                # os.path.join(self.data_dir_raw, "World_Happiness_Report_2018.csv"),
                "filepath": os.path.join(self.data_dir_raw, "World_Happiness_Report_2019.csv"),
                "filename": "World_Happiness_Report_2019.csv",
                "name": "World Happiness Report 2019",
                "description": """
Context: The World Happiness Report is a landmark survey of the state of global happiness. \
The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. \
The World Happiness 2017, which ranks 155 countries by their happiness levels, was released \
at the United Nations at an event celebrating International Day of Happiness on March 20th. \
The report continues to gain global recognition as governments, organizations and civil society \
increasingly use happiness indicators to inform their policy-making decisions. \
Leading experts across fields – economics, psychology, survey analysis, national statistics, \
health, public policy and more – describe how measurements of well-being can be used effectively to \
assess the progress of nations. The reports review the state of happiness in the world today \
and show how the new science of happiness explains personal and national variations in happiness.
Content: The happiness scores and rankings use data from the Gallup World Poll. \
The scores are based on answers to the main life evaluation question asked in the poll. \
This question, known as the Cantril ladder, asks respondents to think of a ladder with \
the best possible life for them being a 10 and the worst possible life being a 0 and to \
rate their own current lives on that scale. The scores are from nationally representative samples \
for the years 2013-2016 and use the Gallup weights to make the estimates representative. \
The columns following the happiness score estimate the extent to which each of six factors – \
economic production, social support, life expectancy, freedom, absence of corruption, and generosity – \
contribute to making life evaluations higher in each country than they are in Dystopia, \
a hypothetical country that has values equal to the world’s lowest national averages for each of the six factors. \
They have no impact on the total score reported for each country, \
but they do explain why some countries rank higher than others.
                """.strip(),
            },
            # {
            #     "id": "11",
            #     "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            #     "filepath": os.path.join(self.data_dir_raw, "Credit_Card_Fraud_Detection.csv"),
            #     "filename": "Credit_Card_Fraud_Detection.csv",
            #     "name": "Credit Card Fraud Detection",
            #     "description": "".strip(),
            # },
            # {
            #     "id": "12",
            #     "url": "https://www.kaggle.com/datasets/START-UMD/gtd",
            #     "filepath": os.path.join(self.data_dir_raw, "Global_Terrorism_Database.csv"),
            #     "filename": "Global_Terrorism_Database.csv",
            #     "name": "Global Terrorism Database",
            #     "description": """
            #             """.strip(),
            # },
        ]
        for ds_dict in self.datasets_info:
            csv_path = ds_dict["filepath"]
            assert os.path.isfile(csv_path), f"Assertion Error: csv file does not exist: {csv_path}"

        self.da_req_high_level = [
            {
                "question": "What is the overall structure and scope of the dataset?",
                "description": "Identify the number of rows, columns, and data types.",
                "visualization": ["a data summary", "table"],
            }
        ]

        self.da_req_low_level = [
            {
                "question": "Which features are strongly correlated?",
                "description": "Calculate correlations between numerical features.",
                "visualization": ["correlation matrix heatmap"],
            }
        ]
        self.da_req_all_list = self.da_req_high_level + self.da_req_low_level
        self.da_req_all_dict = {da_idx: da_req for da_idx, da_req in enumerate(self.da_req_all_list, start=1)}

    def get_da_req(
            self,
            num_req: int = 1,  # TODO: randomly pick num_req data analysis requirements
            da_level: str = "high",
    ) -> List[Dict[str, str]]:
        results = []

        if da_level == "high":
            results.append(self.da_req_high_level[0])
        elif da_level == "low":
            results.append(self.da_req_low_level[0])
        else:
            raise ValueError(f"ValueError: da_level = {da_level}")

        return results


"""
User:
Given a tabular dataset, what questions can we ask to analyze the dataset? \
Present several high-level and lower-level data analysis requirements. \
In addition, for each analysis requirement, choose a suitable chart type for data visualization.

Assistant:
To analyze a tabular dataset, we can ask a series of high-level and lower-level questions \
that help us explore, understand, and interpret the data. Here’s a list of these questions \
along with suggested chart types for each analysis requirement:

### High-Level Data Analysis Requirements

1. **What is the overall structure and scope of the dataset?**
   - **Description:** Identify the number of rows, columns, and data types.
   - **Visualization:** None required, but a data summary or table is often helpful.

2. **What are the key summary statistics?**
   - **Description:** Calculate and visualize measures like mean, median, mode, min, max, and standard deviation for numerical columns.
   - **Chart Type:** Box Plot or Summary Table

3. **Are there any notable trends or patterns in the dataset?**
   - **Description:** Analyze changes in variables over time or other orderings, where applicable.
   - **Chart Type:** Line Chart or Area Chart (for temporal trends), Heatmap (for pairwise comparisons)

4. **Is there missing or inconsistent data?**
   - **Description:** Identify columns or rows with missing values.
   - **Chart Type:** Heatmap (for missing values) or Bar Chart (for percentage of missing values per column)

5. **What is the distribution of key variables?**
   - **Description:** Assess the distribution of numerical and categorical columns.
   - **Chart Type:** Histogram (for numerical variables), Bar Chart or Pie Chart (for categorical variables)

6. **How does the data compare across different segments?**
   - **Description:** Compare groups based on specific characteristics.
   - **Chart Type:** Grouped Bar Chart, Stacked Bar Chart, or Violin Plot (to compare distributions across segments)

---

### Lower-Level Data Analysis Requirements

1. **Which features are strongly correlated?**
   - **Description:** Calculate correlations between numerical features.
   - **Chart Type:** Correlation Matrix Heatmap

2. **Are there any outliers, and how do they affect the data?**
   - **Description:** Identify outliers in numerical columns and evaluate their impact.
   - **Chart Type:** Box Plot or Scatter Plot with outliers highlighted

3. **How does each feature impact the target variable?**
   - **Description:** Explore the relationship between each feature and the target.
   - **Chart Type:** Scatter Plot (for numerical features), Bar Chart or Violin Plot (for categorical features)

4. **Is there any seasonality or periodicity in time-series data?**
   - **Description:** Identify recurring patterns over time.
   - **Chart Type:** Line Chart with multiple time intervals, or Seasonal Decomposition Plot (for advanced analysis)

5. **What are the most common groups or clusters in the data?**
   - **Description:** Use clustering algorithms to find natural groupings.
   - **Chart Type:** Scatter Plot (color-coded by cluster), Dendrogram (for hierarchical clustering)

6. **What are the conditional probabilities or relationships?**
   - **Description:** Calculate conditional probabilities or associations between categories.
   - **Chart Type:** Stacked Bar Chart, Mosaic Plot, or Bubble Chart

7. **How does feature engineering impact model performance?**
   - **Description:** Assess the effect of new or transformed features on the target variable.
   - **Chart Type:** Scatter Plot or Line Plot showing model performance with and without features

8. **What are the implications of data scaling or normalization?**
   - **Description:** Examine the effects of scaling or normalization on features.
   - **Chart Type:** Box Plot (before and after scaling), or Distribution Plot

9. **What are potential biases in the dataset?**
   - **Description:** Identify any demographic or category imbalances.
   - **Chart Type:** Bar Chart (for categorical feature distribution), Density Plot (for numeric distribution)

10. **How does data preprocessing impact outcomes?**
    - **Description:** Compare results before and after cleaning or transformation.
    - **Chart Type:** Side-by-Side Box Plot (before and after transformation), Line Chart (to track impact over steps)

"""
