
## Features

- **Data Upload**: Support for CSV and Excel files
- **Exploratory Data Analysis**: Preview data and get basic information
- **ANOVA Analysis**: Full Two-Way ANOVA statistical tests
- **Assumption Testing**: Checks for normality and homogeneity of variance 
- **Post-hoc Testing**: Tukey HSD for significant factors
- **Effect Size Calculation**: Eta-squared and Partial Eta-squared
- **Rich Visualizations**:
  - Boxplots for main effects
  - Interaction plots
  - Effect size visualizations
  - Heatmaps for interaction effects
- **Export Options**:
  - CSV tables
  - Excel workbook with multiple sheets
  - Word document reports
  - HTML reports
- **Sample Data**: Includes example dataset for demonstration

## Installation

```bash
# Clone the repository
git clone https://github.com/username/TwoWay.git
cd TwoWay

# Install required packages
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file with these dependencies:

```
streamlit
pandas
numpy
statsmodels
matplotlib
seaborn
plotly
scipy
python-docx
openpyxl
```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run TwoWay.py
   ```

2. Open your browser at the URL provided (typically http://localhost:8501)

3. Upload your CSV or Excel file

4. Select your dependent variable and two factors

5. Click "Jalankan Two-Way ANOVA" to perform the analysis

## Example Data Format

Your data should include:
- One continuous dependent variable (numeric)
- Two categorical independent variables (factors)

Example structure:
```
BeratBadan,JenisPakan,BreedSapi
285,Konsentrat,Brahman
290,Konsentrat,Brahman
310,Konsentrat,Limousin
325,Konsentrat,Simental
270,Hijauan,Brahman
298,Hijauan,Limousin
...
```

## Interpreting Results

The application provides comprehensive output to interpret your Two-Way ANOVA results:

1. **ANOVA Table**: Shows test statistics, F-values, and p-values
2. **Effect Sizes**: Indicates the magnitude of effects (small, medium, large)
3. **Post-hoc Tests**: Pairwise comparisons when main effects are significant
4. **Interaction Analysis**: Visualizations to understand how factors interact
5. **Conclusions**: Automatically generated interpretations based on statistical results

## Screenshots

*(Add screenshots of the application here)*

## About

This Two-Way ANOVA Analysis Tool was developed by [Galuh Adi Insani](https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/).

## License

All rights reserved © Galuh Adi Insani