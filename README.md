TwoWay ANOVA Analysis Tool
A Streamlit web application for performing Two-Way ANOVA statistical analysis on your data with comprehensive visualizations and exportable reports.

Overview
This application helps researchers, students, and professionals analyze the influence of two categorical factors on a numerical dependent variable using Two-Way ANOVA. The tool provides interactive visualizations, effect size calculations, post-hoc tests, and exportable reports to aid in statistical analysis and interpretation.

Features
Data Import: Upload CSV or Excel files
Data Validation: Automatic checking for appropriate data types and missing values
ANOVA Analysis: Complete Two-Way ANOVA calculation with interaction effects
Effect Size Metrics: Eta-squared and partial eta-squared calculations
Assumption Testing: Normality and homogeneity of variance checks
Post-hoc Analysis: Tukey HSD tests for significant factors
Interactive Visualizations:
Boxplots and bar charts for main effects
Interaction plots and heatmaps for interaction effects
Effect size visualizations
Exportable Results:
CSV export
Excel reports with multiple sheets
Word document reports
HTML reports
Installation
Clone this repository or download the TwoWay.py file
Install the required dependencies:
Usage
Run the application using Streamlit:

Then open your web browser to the URL displayed in the terminal (typically http://localhost:8501).

Workflow
Upload your CSV or Excel file containing your data
Select your dependent variable (numeric) and two factors (categorical)
Set the alpha level and other options
Click "Run Two-Way ANOVA" to perform the analysis
Explore the results including ANOVA table, effect sizes, and visualizations
Export your results in your preferred format
Example Data
The application includes an example dataset about cattle weight based on feed type and breed to demonstrate Two-Way ANOVA analysis. This example can be downloaded directly from the interface when no file is uploaded.

Requirements
Python 3.6+
Streamlit
Pandas
NumPy
Statsmodels
Matplotlib
Seaborn
Plotly
SciPy
Python-docx
OpenPyXL
Author
Developed by Galuh Adi Insani

License
All rights reserved.