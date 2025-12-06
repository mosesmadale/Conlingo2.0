
#install.packages("readxl")
library(readxl)

setwd("/Users/theeagleeye/Desktop")
getwd()

df <- read_excel("ConlingoAllData.xlsx")
print(df)

regression_model <- lm(CSI_Score_Avg ~ Accuracy_Avg + Tone_Avg + Context_Avg + Empathy_Avg, data = df)

print("--- Full Model Summary (Includes Significance) ---")
summary(regression_model)



#  CALCULATE AND NORMALIZE THE WEIGHTS


# The coefficients (Beta values) are the raw influence weights.
coefficients <- coef(regression_model)

# Extract only the factor coefficients (excluding the Intercept)
factor_coefficients <- coefficients[c("Accuracy_Avg", "Tone_Avg", "Context_Avg", "Empathy_Avg")]

print("--- Raw Beta Coefficients (Influence Weights) ---")
print(factor_coefficients)
print("---------------------------------------------------\n")

# To get normalized weights (which sum to 1), we use the absolute value of the 
# coefficients and normalize them by the sum of all absolute coefficients.

# 1. Take the absolute value of the coefficients
abs_coefficients <- abs(factor_coefficients)

# 2. Calculate the sum of absolute coefficients
sum_abs_coefficients <- sum(abs_coefficients)

# 3. Calculate the normalized weight (the share of the total influence)
normalized_weights <- abs_coefficients / sum_abs_coefficients

print("--- Normalized Factor Weights (Sum to 1) ---")
print(normalized_weights)
print("--------------------------------------------\n")

# Display the results in a cleaner format
results_table <- data.frame(
  Factor = names(normalized_weights),
  Raw_Beta_Estimate = factor_coefficients,
  Normalized_Weight = normalized_weights
)

results_table$Percentage_Weight <- paste0(round(results_table$Normalized_Weight * 100, 2), "%")

print("--- Final Weight Calculation Table ---")
print(results_table)
print("--------------------------------------\n")

# 
#  FINAL SCORING FORMULA


cat("\n--- Proposed Scoring Formula (Based on Weights) ---\n")

# Construct the final formula string
formula_parts <- paste(
  round(normalized_weights, 3), 
  names(normalized_weights), 
  sep = " * ", 
  collapse = " + "
)

final_formula <- paste("Predicted CSI Score = (", formula_parts, ")")
cat(final_formula, "\n\n")

