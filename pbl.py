import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import display, Markdown

# =====================================================================
# 1. PREPARE NUMERICAL SOLVER CALCULATIONS (DYNAMICALLY)
# =====================================================================
scaler = model_pipeline.named_steps['scaler']
regressor = model_pipeline.named_steps['regressor']

means = scaler.mean_
scales = scaler.scale_
intercept = regressor.intercept_
coefficients = regressor.coef_

sample_features = np.array([over_budget_sample[f] for f in features])

# Detailed LaTeX Math Formulation printout
math_markdown = """### Detailed Mathematical Solution Walkthrough\n\n"""
math_markdown += "The Linear Regression model solves the following mathematical equation for a standardized input vector $X$: \n"
math_markdown += "$$\\hat{y} = \\beta_0 + \\sum_{i=1}^{n} \\beta_i \\cdot z_i$$\n\n"
math_markdown += "Where the standardized value $z_i$ is computed as: \n"
math_markdown += "$$z_i = \\frac{x_i - \\mu_i}{\\sigma_i}$$$\n\n"
math_markdown += "**Model Calculation Breakdown Table:**\n\n"
math_markdown += "| Feature ($i$) | Raw Value ($x_i$) | Mean ($\\mu_i$) | Std Dev ($\\sigma_i$) | Scaled Value ($z_i$) | Coefficient ($\\beta_i$) | Contribution ($\\beta_i z_i$) |\n"
math_markdown += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"

running_sum = intercept
numerical_steps = []

for i, feature_name in enumerate(features):
    x_val = sample_features[i]
    mu = means[i]
    sigma = scales[i]
    beta = coefficients[i]

    x_scaled = (x_val - mu) / sigma
    contribution = beta * x_scaled
    running_sum += contribution

    math_markdown += f"| **{feature_name.replace('_', ' ').title()}** | {x_val:,.2f} | {mu:,.2f} | {sigma:,.2f} | {x_scaled:+.4f} | {beta:+,.2f} | **{contribution:+,.2f}** |\n"
    numerical_steps.append(f"{feature_name[:10].title()}: ({x_val:.0f} - {mu:.0f})/{sigma:.0f} * {beta:.1f} = {contribution:+.1f}")

pipeline_prediction = model_pipeline.predict(pd.DataFrame([over_budget_sample])[features])[0]

math_markdown += f"\n**Final Arithmetic Assembly:**\n"
math_markdown += f"$$\\hat{{y}} = {intercept:,.2f} "
for i, feature_name in enumerate(features):
    x_val = sample_features[i]
    x_scaled = (x_val - means[i]) / scales[i]
    contrib = coefficients[i] * x_scaled
    math_markdown += f" + ({contrib:+,.2f})"
math_markdown += "$$\n"
math_markdown += f"* **Manual Solved Prediction:** `{running_sum:,.2f} INR`\n"
math_markdown += f"* **Model Predict() Output:** `{pipeline_prediction:,.2f} INR`\n"
math_markdown += f"* **Absolute Error Margin:** `{abs(running_sum - pipeline_prediction):.2e} INR`\n"

display(Markdown(math_markdown))

# =====================================================================
# 2. GENERATE COMPLETE DASHBOARD WITH GRAPHICS & NUMERICS
# =====================================================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

categories = ['rent', 'groceries', 'dining_out', 'transport', 'utilities', 'loan_emi']
avg_spend = df[categories].mean()

original_vals = [over_budget_sample['rent'], over_budget_sample['dining_out'], over_budget_sample['transport']]
optimized_vals = [8872, 0, 2275]
labels_opt = ['Rent', 'Dining Out', 'Transport']

fig, axes = plt.subplots(3, 2, figsize=(20, 22))
fig.suptitle('Unified Personal Finance Dashboard & Numerical Solver', fontsize=22, fontweight='bold', y=0.97, color='#1a1a1a')

# --- 1. Expense Breakdown (Top Left) ---
axes[0, 0].pie(avg_spend, labels=[c.replace('_', ' ').title() for c in categories], autopct='%1.1f%%',
              startangle=140, colors=sns.color_palette("Blues_r", len(categories)),
              wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True})
axes[0, 0].set_title('Average Monthly Expense Composition', fontsize=14, fontweight='bold', pad=15)

# --- 2. Feature Impact on Total Expenses (Top Right) ---
sns.barplot(
    x='Standardized Coefficient',
    y='Feature',
    data=importance_df,
    palette='vlag',
    ax=axes[0, 1],
    hue='Feature',
    legend=False
)
axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_title('Feature Impact (Model Coefficients)', fontsize=14, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Coefficient Value', fontsize=12)
axes[0, 1].set_ylabel('Features', fontsize=12)

# --- 3. Budget Adherence Status (Middle Left) ---
over_budget_counts = df['over_budget'].value_counts()
axes[1, 0].pie(over_budget_counts, labels=['Within Budget', 'Over Budget'], autopct='%1.1f%%',
              startangle=90, colors=['#2ecc71', '#e74c3c'],
              wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1.5))
axes[1, 0].set_title('User Cohort Budget Adherence', fontsize=14, fontweight='bold', pad=15)

# --- 4. Income vs. Total Expense (Middle Right) ---
sns.scatterplot(
    x='monthly_income',
    y='total_expense',
    hue='over_budget',
    palette={0: '#2ecc71', 1: '#e74c3c'},
    alpha=0.6,
    data=df,
    ax=axes[1, 1]
)
sns.regplot(
    x='monthly_income',
    y='total_expense',
    data=df,
    scatter=False,
    ax=axes[1, 1],
    color='#2c3e50',
    line_kws={'linestyle': '--', 'linewidth': 2}
)
axes[1, 1].set_title('Income vs. Expenses (Color by Budget Violation)', fontsize=14, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Monthly Income (INR)', fontsize=12)
axes[1, 1].set_ylabel('Total Expense (INR)', fontsize=12)
axes[1, 1].legend(title='Over Budget status', labels=['Within Budget', 'Over Budget'])

# --- 5. Solver Optimization Results Comparison (Bottom Left) ---
x_indexes = np.arange(len(labels_opt))
width = 0.35
axes[2, 0].bar(x_indexes - width/2, original_vals, width, label='Original Spending', color='#e74c3c', alpha=0.85)
axes[2, 0].bar(x_indexes + width/2, optimized_vals, width, label='Optimized Spending', color='#2ecc71', alpha=0.85)
axes[2, 0].set_xticks(x_indexes)
axes[2, 0].set_xticklabels(labels_opt)
axes[2, 0].set_title('Numerical Solver Cuts for Sample User', fontsize=14, fontweight='bold', pad=15)
axes[2, 0].set_ylabel('Amount (INR)', fontsize=12)
axes[2, 0].legend()

# --- 6. Step-by-Step Numerical Formula Solving Panel (Bottom Right) ---
axes[2, 1].axis('off')
axes[2, 1].text(0.01, 0.95, 'Linear Regression Numerical Solver Steps:', fontsize=15, fontweight='bold', color='#1a1a1a')
axes[2, 1].text(0.01, 0.90, f"Base Intercept (Beta_0): {intercept:.1f}", fontsize=11, fontweight='bold', color='#2c3e50')

y_pos = 0.83
for step_line in numerical_steps:
    axes[2, 1].text(0.01, y_pos, f"• {step_line}", fontsize=10, family='monospace', color='#333333')
    y_pos -= 0.05

axes[2, 1].text(0.01, y_pos, f"Manual Combined Calculation:  {running_sum:.2f} INR", fontsize=11, fontweight='bold', color='#27ae60')
axes[2, 1].text(0.01, y_pos - 0.04, f"Model Pipeline Output:        {pipeline_prediction:.2f} INR", fontsize=11, fontweight='bold', color='#2980b9')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
