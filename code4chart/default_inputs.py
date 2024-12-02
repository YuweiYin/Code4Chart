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
        self.data_dir = os.path.join(project_root_dir, "data/autoda")
        # self.ckpt_dir = os.path.join(project_root_dir, "ckpt/autoda")
        os.makedirs(self.data_dir, exist_ok=True)
        # os.makedirs(self.ckpt_dir, exist_ok=True)

        # Tabular datasets:
        # https://www.kaggle.com/datasets?sort=votes
        # // https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        self.data_csv_list = [
            # https://www.kaggle.com/datasets/imdevskp/corona-virus-report
            os.path.join(self.data_dir, "COVID-19_Dataset.csv"),
            # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
            # os.path.join("Credit_Card_Fraud_Detection.csv"),
            # https://www.kaggle.com/datasets/START-UMD/gtd
            os.path.join(self.data_dir, "Global_Terrorism_Database.csv"),
            # https://www.kaggle.com/datasets/uciml/iris
            os.path.join(self.data_dir, "Iris_Species.csv"),
            # https://www.kaggle.com/datasets/shivamb/netflix-shows
            os.path.join(self.data_dir, "Netflix_Movies_and_TV_Shows.csv"),
            # https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
            os.path.join(self.data_dir, "New_York_City_Airbnb_Open_Data_2019.csv"),
            # https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
            os.path.join(self.data_dir, "Stroke_Prediction_Dataset.csv"),
            # https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
            os.path.join(self.data_dir, "Students_Performance_in_Exams.csv"),
            # https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016
            os.path.join(self.data_dir, "Suicide_Rates_Overview_1985_to_2016.csv"),
            # https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
            os.path.join(self.data_dir, "TMDB_5000_Movie_Dataset.csv"),
            # https://www.kaggle.com/datasets/unsdsn/world-happiness
            # os.path.join(self.data_dir, "World_Happiness_Report_2015.csv"),
            # os.path.join(self.data_dir, "World_Happiness_Report_2016.csv"),
            # os.path.join(self.data_dir, "World_Happiness_Report_2017.csv"),
            # os.path.join(self.data_dir, "World_Happiness_Report_2018.csv"),
            os.path.join(self.data_dir, "World_Happiness_Report_2019.csv"),
        ]
        for csv_path in self.data_csv_list:
            assert os.path.isfile(csv_path), f"Assertion Error: csv file does not exist: {csv_path}"
        self.data_csv_dict = {csv_idx: csv_path for csv_idx, csv_path in enumerate(self.data_csv_list, start=1)}
        self.data_csv_names = [csv_path.split("/")[-1].split(".")[0].replace("_", " ").strip()
                               for csv_path in self.data_csv_list]

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
