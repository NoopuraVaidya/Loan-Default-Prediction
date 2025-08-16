# Loan Default Prediction

This project analyzes loan default risk using **Logistic Regression** and **CART (Decision Tree)** models.  
It evaluates which customer attributes (FICO score, income, number of households, age, gender, loan term, etc.) are significant in predicting loan defaults.

## Project Structure
- `notebooks/` → Jupyter Notebook with full analysis  
- `report/` → Project report in DOCX/PDF format  
- `README.md` → Project description

## Techniques Used
- Logistic Regression  
- Decision Tree (CART)  
- Predictor Importance Analysis  
- Confusion Matrix Evaluation  

## Performance Metrics
The models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F-measure**

## Dataset
The dataset includes customer financial and demographic variables such as:
- FICO score  
- Income  
- Number of household members  
- Loan amount and term  
- Homeownership status  
- Gender  
- Default status  

## Results
- Logistic Regression achieved **~96% accuracy** with strong precision and recall.  
- CART achieved **near-perfect accuracy**, though with a risk of overfitting.  

## Ethical Considerations
- Gender appears statistically significant in logistic regression, raising fairness concerns.  
- CART did not include gender as a predictor, suggesting less bias.  
- Future work should focus on fairness-aware modeling.  

---
