
# Bank Risk Controller System

This repository contains a Python script designed for a bank risk controller system. The script utilizes a pre-trained decision tree model to assess and recommend services based on customer attributes. The system is built using Streamlit for the user interface and incorporates several libraries for data processing and machine learning.

## Features

- **Load and Use a Pre-trained Model**: The script loads a decision tree model from a pickle file for making predictions.
- **Categorical Data Encoding**: Encodes user input using label encoders for various categorical features.
- **Streamlit User Interface**: Provides a user-friendly interface for selecting customer attributes and generating recommendations.
- **Service Recommendations**: Generates and displays top service recommendations based on user input.

## Prerequisites

To run this script, you need the following libraries installed:

- pandas
- numpy
- pickle
- streamlit
- sqlite3
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- streamlit-option-menu

You can install the required libraries using pip:

```sh
pip install pandas numpy pickle-mixin streamlit sqlite3 matplotlib seaborn scikit-learn tensorflow streamlit-option-menu
```

## Usage

1. **Load the Trained Model**:
   The script loads a pre-trained decision tree model from a pickle file located at `/content/DTmodel21.pkl`.

2. **Initialize Label Encoders**:
   Label encoders are initialized and fitted with possible categorical values for encoding user input.

3. **Streamlit User Interface**:
   The user interface is created using Streamlit, allowing users to select:
   - Domain
   - Customer
   - Customer Size Revenue
   - Packages

4. **Generate Recommendations**:
   Based on the selected inputs, the script filters the data and generates top service recommendations using the pre-trained model.

5. **Display Recommendations**:
   The top service recommendations are displayed on the Streamlit interface.

## Running the Script

To run the script, use the following command:

```sh
streamlit run final.py
```

This will start a local web server and open the Streamlit user interface in your default web browser.

## Example

After running the script, you can select the domain, customer, revenue size, and packages from the sidebar. Click the "Recommend Services" button to generate and view the top service recommendations.

## Bank Risk Controller System

### Overview

The Bank Risk Controller System is designed to help banks assess risk and recommend suitable services to their customers. It leverages machine learning techniques to analyze customer data and provide tailored recommendations, improving customer satisfaction and risk management.

### Functionalities

- **Risk Assessment**: Evaluates customer risk based on provided attributes.
- **Service Recommendation**: Suggests appropriate banking services to mitigate identified risks.
- **Data Visualization**: Provides graphical representations of data and recommendations for better understanding.

### How It Works

1. **Input Customer Data**: Users input customer data through the Streamlit interface.
2. **Risk Analysis**: The system analyzes the data using the pre-trained model.
3. **Recommendation Generation**: Based on the analysis, the system generates and displays recommended services.

### Benefits

- **Enhanced Risk Management**: Helps banks identify and manage risks more effectively.
- **Improved Customer Satisfaction**: Provides personalized service recommendations to meet customer needs.
- **Data-Driven Decisions**: Utilizes machine learning to support informed decision-making.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://www.streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [Scikit-Learn](https://scikit-learn.org/)

