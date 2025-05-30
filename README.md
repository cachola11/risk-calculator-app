# Risk Calculator App

## Overview
The Risk Calculator App is a financial risk management tool designed to calculate and visualize Value at Risk (VaR) and Expected Shortfall (ES) metrics. It combines historical and parametric approaches to provide comprehensive insights into potential financial losses under various market conditions.

## Features
- **Historical VaR Calculation**: Calculate the historical Value at Risk for financial returns using user-uploaded data.
- **Parametric VaR and ES**: Utilize the `RiskMetricsCalculator` class to compute VaR and ES using normal and Student's t-distributions.
- **Data Visualization**: Visualize return distributions, VaR, and ES metrics using Matplotlib and Streamlit.
- **User-Friendly Interface**: Streamlit provides an interactive web interface for users to upload data and view results.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd risk-calculator-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
streamlit run src/VaR_Historico.py
```

### Input Data
Users can upload CSV or Excel files containing historical price data. The application will process the data to calculate returns and subsequently compute the VaR and ES metrics.

### Example
1. Upload a CSV or Excel file with historical price data.
2. Select the appropriate columns for dates, funds, and benchmarks.
3. Choose the confidence level for VaR calculations.
4. View the calculated VaR and ES metrics along with visualizations of the return distributions.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.