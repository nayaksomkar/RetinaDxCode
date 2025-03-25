import pandas as pd

filepath = r"C:\Users\nayak\Downloads\Dataset B\data.xlsx"
# Load CSV filed
df = pd.read_excel(filepath) 
 # Display first 5 rows
print(df.head())

# Select the columns you want to write to the new file
selected_columns = ["ID", "Left-Fundus", "Left-Diagnostic Keywords", "N", "D", "G", "C", "A", "H", "M", "O"] 
# Create a new DataFrame with selected columns
new_df = df[selected_columns] 
# Save to a new Excel file
new_df.to_excel("LeftFundusDSB.xlsx", index=False, engine="openpyxl")
print("New file created successfully!")

selected_columns = ["ID", "Right-Fundus", "Right-Diagnostic Keywords", "N", "D", "G", "C", "A", "H", "M", "O"]  
new_df = df[selected_columns]
# Save to a new Excel file
new_df.to_excel("RightFundusDSB.xlsx", index=False, engine="openpyxl")

print("New file created successfully!")

